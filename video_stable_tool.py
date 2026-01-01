# pip install opencv-python numpy tqdm
# python video_stable_tool.py

import cv2
import numpy as np
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, ttk
import threading

# 全局变量
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

def get_bbox_coords(mask):
    """从二值化图像中获取矩形边界框坐标"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return x, y, x + w, y + h

def expand_bbox(bbox, scale_x, aspect_ratio, frame_w, frame_h):
    """扩展边界框并保持长宽比"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    width = (x2 - x1) * scale_x
    height = width * aspect_ratio
    
    half_w, half_h = width / 2, height / 2
    new_x1 = max(0, center_x - half_w)
    new_y1 = max(0, center_y - half_h)
    new_x2 = min(frame_w, center_x + half_w)
    new_y2 = min(frame_h, center_y + half_h)
    
    # 如果超出边界则平移
    if new_x2 - new_x1 < width:
        diff = width - (new_x2 - new_x1)
        new_x1 = max(0, new_x1 - diff / 2)
        new_x2 = min(frame_w, new_x1 + width)
        new_x1 = max(0, new_y2 - width)
    if new_y2 - new_y1 < height:
        diff = height - (new_y2 - new_y1)
        new_y1 = max(0, new_y1 - diff / 2)
        new_y2 = min(frame_h, new_y1 + height)
        new_y1 = max(0, new_y2 - height)
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def process_videos(ref_path, target_path, output_path, expand_x, progress_callback=None):
    ref_cap = cv2.VideoCapture(ref_path)
    target_cap = cv2.VideoCapture(target_path)
    
    # 获取视频信息
    ref_width = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ref_height = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_width = int(target_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    target_height = int(target_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = target_cap.get(cv2.CAP_PROP_FPS)
    
    # 计算分辨率比例
    scale_x = target_width / ref_width
    scale_y = target_height / ref_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))
    
    total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(total_frames)):
        ret1, ref_frame = ref_cap.read()
        ret2, target_frame = target_cap.read()
        
        if not (ret1 and ret2):
            break
            
        # 二值化参考帧
        gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_ref, 127, 255, cv2.THRESH_BINARY)
        
        # 获取矩形坐标
        bbox = get_bbox_coords(binary)
        if bbox is None:
            # 如果没有检测到矩形，输出黑色帧
            out.write(np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8))
            continue
        
        # 变换到目标视频坐标系
        x1, y1, x2, y2 = bbox
        scaled_bbox = (
            int(x1 * scale_x), 
            int(y1 * scale_y),
            int(x2 * scale_x), 
            int(y2 * scale_y)
        )
        
        # 扩展区域
        final_bbox = expand_bbox(
            scaled_bbox, 
            expand_x, 
            TARGET_HEIGHT / TARGET_WIDTH, 
            target_width, 
            target_height
        )
        
        # 切割并调整大小
        x1, y1, x2, y2 = final_bbox
        cropped = target_frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT))
        
        out.write(resized)
        
        if progress_callback:
            progress_callback((i + 1) / total_frames * 100)
    
    ref_cap.release()
    target_cap.release()
    out.release()

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频处理工具")
        
        # 变量
        self.ref_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.processing = False

        self.expandx = tk.DoubleVar(value=2)

        # 分辨率选项
        self.resolution_options = {
            "8K": (7680, 4320),
            "4K": (3840, 2160),
            "2K": (2560, 1440),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480),
            "360p": (640, 360),
            "240p": (426, 240)
        }
        
        # 创建界面
        self.create_widgets()

    def update_resolution(self, combo):
        global TARGET_WIDTH, TARGET_HEIGHT
        width, height = self.resolution_options[combo.get()]
        TARGET_WIDTH = width
        TARGET_HEIGHT = height

    
    def create_widgets(self):
        # 参考视频
        tk.Label(self.root, text="mocha输出追踪视频:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.ref_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.root, text="浏览", command=self.browse_ref).grid(row=0, column=2, padx=5, pady=5)
        
        # 目标视频
        tk.Label(self.root, text="源视频:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.target_path, width=40).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(self.root, text="浏览", command=self.browse_target).grid(row=1, column=2, padx=5, pady=5)
        
        # 输出路径
        tk.Label(self.root, text="输出路径:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        tk.Entry(self.root, textvariable=self.output_path, width=40).grid(row=2, column=1, padx=5, pady=5)
        tk.Button(self.root, text="浏览", command=self.browse_output).grid(row=2, column=2, padx=5, pady=5)

        # 分辨率选择
        tk.Label(self.root, text="输出分辨率:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        resolution_combo = ttk.Combobox(self.root, values=list(self.resolution_options.keys()), state="readonly", width=15, textvariable=tk.StringVar(value="1080p"))
        resolution_combo.bind("<<ComboboxSelected>>", lambda e: self.update_resolution(resolution_combo))
        resolution_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        resolution_combo.set("1080p")

        # 扩展倍数
        tk.Label(self.root, text="扩展倍数:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        tk.Spinbox(self.root, from_=0.1, to=10.0, increment=2, textvariable=self.expandx, width=10).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # 进度条
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, padx=5, pady=10)
        
        # 状态标签
        self.status_label = tk.Label(self.root, text="就绪")
        self.status_label.grid(row=6, column=0, columnspan=3, padx=5, pady=5)
        
        # 处理按钮
        self.process_btn = tk.Button(self.root, text="开始处理", command=self.start_processing)
        self.process_btn.grid(row=7, column=0, columnspan=3, padx=5, pady=10)

        # 使用说明文本框
        usage_frame = tk.LabelFrame(self.root, text="使用说明")
        usage_frame.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        usage_text = tk.Text(usage_frame, height=6, width=60, wrap=tk.WORD, state="normal")
        usage_text.insert("1.0", 
                          """
mocha pro程序里，参考我的上一个视频完成追踪后，点左上角file--export rendered clip--export to改为QuickTime movie--OK--resolution可以调低，别的参数也可以调低--导出
不过mocha默认不会降分辨率，推荐先把视频分辨率缩小到1/8以下再进行追踪标注，提升计算速度。以上步骤分辨率变换都不影响什么，取的是相对位置，输出视频是一个黑色的、包含目标白色矩形的视频。
然后运行我给的python文件即可，设置目标分辨率、外扩倍数（为1则按照你绘制的追踪矩形，否则进行中心不变的 外扩，标准为宽度，然后按照输出长宽比得到目标roi），然后就能生成目标视频
然后可通过以下命令进行音频复制和视频压缩，或者你自己导到别的视频编辑软件操作
ffmpeg -i 稳定视频.mp4 -i 源视频.MP4 -c:v hevc_nvenc -b:v 10M -c:a aac -b:a 128k -map 0:v:0 -map 1:a:0 最终视频.mp4 （适用于N卡、1080P视频，若有其他情况可提供给AI让修改。需要安装ffmpeg）

本项目链接：https://github.com/Xxianna/video-stable-tool
bilibili示例视频链接：https://www.bilibili.com/BV15MvDBXE5c
                          """)
        usage_text.config(state="disabled")
        usage_text.pack(padx=5, pady=5)

    
    def browse_ref(self):
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.mov *.avi")])
        if file_path:
            self.ref_path.set(file_path)
    
    def browse_target(self):
        file_path = filedialog.askopenfilename(filetypes=[("视频文件", "*.mp4 *.mov *.avi")])
        if file_path:
            self.target_path.set(file_path)
    
    def browse_output(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4文件", "*.mp4")])
        if file_path:
            self.output_path.set(file_path)
    
    def update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()
    
    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update_idletasks()
    
    def start_processing(self):
        if not self.ref_path.get() or not self.target_path.get() or not self.output_path.get():
            tk.messagebox.showerror("错误", "请填写所有文件路径")
            return
        
        self.processing = True
        self.process_btn.config(state="disabled")
        threading.Thread(target=self.process_video_thread, daemon=True).start()
    
    def process_video_thread(self):
        try:
            self.update_status("正在处理视频...")
            process_videos(
                self.ref_path.get(), 
                self.target_path.get(), 
                self.output_path.get(),
                self.expandx,
                progress_callback=self.update_progress
            )
            self.update_status("处理完成!")
            tk.messagebox.showinfo("完成", "视频处理完成!")
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            tk.messagebox.showerror("错误", f"处理失败: {str(e)}")
        finally:
            self.processing = False
            self.process_btn.config(state="normal")
            self.progress['value'] = 0

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()
