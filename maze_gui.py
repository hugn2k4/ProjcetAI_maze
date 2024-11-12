import math
import threading
from tkinter import ttk

from simpleai.search import SearchProblem, astar
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
import random

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.7

# Create the cost dictionary
COSTS = {
    "up": cost_regular,
    "down": cost_regular,
    "left": cost_regular,
    "right": cost_regular,
    "up left": cost_diagonal,
    "up right": cost_diagonal,
    "down left": cost_diagonal,
    "down right": cost_diagonal,
}

MAP = """
############################################################
#         #               #                   #            #
# ####    ########        #        ######     #            #
#    #    #               #        #          #            #
#    ###     #####  ######         #######    ######       #
#      #   ###   #                    #               ######
#      #     #   #  #  #   ###        #  #  #              #
#     #####    #    #  #      #  ######    #######    #### #
#              #        #######                #      #    #
# ####    ###       #########      ######      #      ######
#    #        ##                  #            #           #
#    ###    ###    ##########     #   ##########    ###### #
#      #     ###   ##             #            ##          #
#   ####     #     ##    #######  #   ##########           #
#             #     #             #   #                    #
#             ##    ##            #    ######              #
#              ##     ##                 ##     ########## #
#      ######   ##     ##      #######    ##      ##       #
#              ##            ##           ##               #
############################################################
#         #               #                  #             #
#         #               #                   #            #
# ####    ########        #        ######     #            #
#    #    #               #        #          #            #
#    ###     #####  ######         #######    ######       #
#      #   ###   #                    #               ######
#      #     #   #  #  #   ###        #  #  #              #
#     #####    #    #  #      #  ######    #######    #### #
#              #        #######                #      #    #
# ####    ###       #########      ######      #      ######
#    #        ##                  #            #           #
#    ###    ###    ##########     #   ##########    ###### #
#      #     ###   ##             #            ##          #
#   ####     #     ##    #######  #   ##########           #
#             #     #             #   #                    #
#             ##    ##            #    ######              #
#              ##     ##                 ##     ########## #
#      ######   ##     ##      #######    ##      ##       #
#              ##            ##           ##               #
############################################################
"""

# Convert map to a list
MAP = [list(x) for x in MAP.split("\n") if x]
M = 40
N = 60
W = 21 # giảm w nếu lớn quá
# Màu sắc cho mê cung
mau_tuong = np.zeros((W, W, 3), np.uint8) + (82, 110, 72)
mau_nen = np.zeros((W, W, 3), np.uint8) + (194, 255, 199)
image = np.ones((M*W, N*W, 3), np.uint8)*255

for x in range(0, M):
    for y in range(0, N):
        if MAP[x][y] == '#':
            image[x*W:(x+1)*W, y*W:(y+1)*W] = mau_tuong
        elif MAP[x][y] == ' ':
            image[x*W:(x+1)*W, y*W:(y+1)*W] = mau_nen

color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(color_coverted)

def random_map(height, width, wall_density):
    # Tạo một bản đồ với viền tường
    new_map = [["#"] * width for _ in range(height)]
    # Thay thế các ô bên trong bằng tường hoặc không gian trống dựa vào wall_density
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if random.random() < wall_density:
                new_map[y][x] = "#"  # Tạo tường
            else:
                new_map[y][x] = " "  # Không gian trống
    return new_map
# Class containing the methods to solve the maze
class MazeSolver(SearchProblem):
    # Initialize the class
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(MazeSolver, self).__init__(initial_state=self.initial)

    # Define the method that takes actions
    # to arrive at the solution
    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)

        return actions

    # Update the state based on the action
    def result(self, state, action):
        x, y = state

        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1

        new_state = (x, y)

        return new_state

    # Check if we have reached the goal
    def is_goal(self, state):
        return state == self.goal

    # Compute the cost of taking an action
    def cost(self, state, action, state2):
        return COSTS[action]

    # Heuristic that we use to arrive at the solution
    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal

        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.dem = 0
        self.path_found = False  # Biến trạng thái kiểm tra đã tìm đường
        self.size =3
        self.level = 0.3
        self.title('Giải Mê Cung')

        # Màu sắc chủ đạo cho ứng dụng
        self.bg_color = "#C4E1F6"  # Màu nền chính của ứng dụng
        self.menu_bg_color = "#7AB2D3"  # Màu nền cho menu
        self.button_bg_color = "#4A628A"  # Màu nền cho nút
        self.button_fg_color = "#f8f8f2"  # Màu chữ trên nút
        self.canvas_bg_color = "#DFF2EB"  # Màu nền cho canvas
        self.status_fg_color = "#ff5555"  # Màu chữ cho trạng thái

        # Cấu hình Style cho các nút
        style = ttk.Style()
        style.configure("RoundedButton.TButton",
                        relief="flat",  # Mở bỏ viền nổi bật
                        background="#4CAF50",  # Màu nền nút
                        foreground="black",  # Màu chữ
                        font=("Arial", 12, "bold"),
                        padding=10)

        # Thiết lập hiệu ứng khi di chuột
        style.map("RoundedButton.TButton",
                  background=[("active", "#45a049"), ("pressed", "#388e3c")])  # Màu khi hover

        # Thiết lập màu nền cho cửa sổ chính
        self.configure(bg=self.bg_color)
        # Tạo Canvas hiển thị mê cung với màu nền
        self.cvs_me_cung = tk.Canvas(self, width=N * W, height=M * W, relief=tk.SUNKEN, border=1,
                                     bg=self.canvas_bg_color)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.cvs_me_cung.bind("<Button-1>", self.xu_ly_mouse)

        # Tạo frame chứa các nút điều khiển với màu nền nổi bật và viền
        lbl_frm_menu = tk.LabelFrame(self, text="BẢNG ĐIỀU KHIỂN", padx=10, pady=10,
                                     font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#72BF78")
        lbl_frm_menu.pack(pady=20, padx=20, fill="both", expand=True)
        # Thiết lập khung bo cong cho lbl_frm_menu
        lbl_frm_menu.config(borderwidth=2, relief="solid",bg="#BFECFF")
        lbl_frm_menu.place(x=50, y=50, width=300, height=270)

        # Các nút điều khiển với màu sắc

        btn_find_path = ttk.Button(lbl_frm_menu, text='Tìm đường', style="RoundedButton.TButton", command=self.btn_find_path_click)
        btn_start = ttk.Button(lbl_frm_menu, text='Bắt đầu', style="RoundedButton.TButton", command=self.btn_start_click)
        btn_reset = ttk.Button(lbl_frm_menu, text='Làm mới', style="RoundedButton.TButton", command=self.btn_reset_click)
        btn_random_map = ttk.Button(lbl_frm_menu, text='Bản đồ ngẫu nhiên', style="RoundedButton.TButton", command=self.btn_random_map_click)

        # chỉnh sửa kích thước
        btn_mini_map = ttk.Button(lbl_frm_menu, text='10x30', style="RoundedButton.TButton", command=self.btn_mini_map_click)
        btn_medium_map = ttk.Button(lbl_frm_menu, text='20x60', style="RoundedButton.TButton", command=self.btn_medium_map_click)
        btn_large_map = ttk.Button(lbl_frm_menu, text='40x60', style="RoundedButton.TButton", command=self.btn_large_map_click)

        # độ phức tạp mê cung
        btn_easy_level = ttk.Button(lbl_frm_menu, text='Đơn giản', style="RoundedButton.TButton", command=self.btn_easy_level_click)
        btn_medium_level = ttk.Button(lbl_frm_menu, text='Bình thường', style="RoundedButton.TButton", command=self.btn_medium_level_click)
        btn_hard_level = ttk.Button(lbl_frm_menu, text='Phức tạp', style="RoundedButton.TButton", command=self.btn_hard_level_click)

        # Đặt các nút vào LabelFrame và căn giữa
        btn_find_path.grid(row=0, column=0, pady=10, padx=10, sticky="ew", columnspan=3)
        btn_start.grid(row=1, column=0, columnspan=2, pady=10, padx=10)
        btn_reset.grid(row=1, column=1, columnspan=2, pady=10, padx=10)
        btn_random_map.grid(row=4, column=0, columnspan=3, pady=10, padx=10, sticky="ew")

        btn_mini_map.grid(row=2, column=0, pady=10, padx=10, sticky="ew")
        btn_medium_map.grid(row=2, column=1, pady=10, padx=10, sticky="ew")
        btn_large_map.grid(row=2, column=2, pady=10, padx=10, sticky="ew")

        btn_easy_level.grid(row=3, column=0, pady=10, padx=10, sticky="ew")
        btn_medium_level.grid(row=3, column=1, pady=10, padx=10, sticky="ew")
        btn_hard_level.grid(row=3, column=2, pady=10, padx=10, sticky="ew")

        # Nhãn trạng thái với màu sắc mới
        self.lbl_status = tk.Label(lbl_frm_menu, text="Thông báo!", fg=self.status_fg_color, font=("Arial", 12, "bold"))
        self.lbl_status.grid(row=5, column=0, columnspan=3, pady=10, padx=10, sticky="ew")

        # Nhãn mức level
        self.lbl_level = tk.Label(lbl_frm_menu, text="Mặc định!", fg=self.status_fg_color, font=("Arial", 12, "bold"))
        self.lbl_level.grid(row=6, column=0, columnspan=3, pady=10, padx=10, sticky="ew")

        # Đặt các thành phần lên cửa sổ chính
        self.cvs_me_cung.grid(row=0, column=0, padx=10, pady=10)
        lbl_frm_menu.grid(row=0, column=1, padx=10, pady=10, sticky=tk.N)

        # Đặt grid layout để tự động thay đổi kích thước
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

    def update_map_image(self):
        if self.size == 1:
            M = 10
            N = 30
        elif self.size == 2:
            M = 20
            N = 60
        else:
            M = 40
            N = 60
        image = np.ones((M * W, N * W, 3), np.uint8) * 255
        global MAP, pil_image
        MAP = random_map(M, N,self.level)
        for x in range(M):
            for y in range(N):
                if MAP[x][y] == '#':
                    image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_tuong
                else:
                    image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_nen
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)

    def xu_ly_mouse(self, event):
        px, py = event.x, event.y
        x, y = px // W, py // W
        if MAP[y][x] == '#':
            return

        if self.dem == 0:
            MAP[y][x] = 'o'
            # Vẽ điểm bắt đầu bằng hình tròn màu xanh dương (có kích thước lớn hơn)
            self.cvs_me_cung.create_oval(x * W + W // 6, y * W + W // 6, (x + 1) * W - W // 6, (y + 1) * W - W // 6,
                                         outline='#5AB2FF', fill='#5AB2FF')  # Màu xanh dương
            self.dem += 1
        elif self.dem == 1:
            MAP[y][x] = 'x'
            # Vẽ điểm kết thúc bằng hình vuông màu xanh lá
            self.cvs_me_cung.create_rectangle(x * W + 2, y * W + 2, (x + 1) * W - 2, (y + 1) * W - 2,
                                              outline='#F875AA', fill='#F875AA')  # Màu xanh lá
            self.dem += 1

    def btn_start_click(self):
        self.lbl_status.config(text="Thông báo!")
        if not self.path_found:
            self.lbl_status.config(text="Hãy tìm đường!")
        else:
            problem = MazeSolver(MAP)
            result = astar(problem, graph_search=True)
            if result is None:
                self.lbl_status.config(text="Không tìm thấy đường đi!")
            else:
                path = [x[1] for x in result.path()]
                for i in range(1, len(path)- 1):
                    x, y = path[i]
                    self.cvs_me_cung.create_oval(x * W + W // 4, y * W + W // 4, (x + 1) * W - W // 4,(y + 1) * W - W // 4,
                                                 outline='#ff5555', fill='#ff5555')
                    time.sleep(0.1)
                    self.cvs_me_cung.update()
                self.lbl_status.config(text="Thành công!")

    def btn_reset_click(self):
        global MAP
        M = len(MAP)  # Số hàng trong MAP
        N = len(MAP[0]) if M > 0 else 0  # Số cột trong MAP (kiểm tra MAP có ít nhất 1 hàng)
        self.cvs_me_cung.delete(tk.ALL)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.dem = 0
        self.path_found = False  # Đặt lại trạng thái tìm đường
        self.lbl_status.config(text="Làm mới thành công!")
        for x in range(0, M):
            for y in range(0, N):
                if MAP[x][y] == 'o' or MAP[x][y] == 'x':
                    MAP[x][y] = ' '

    def update_map_size(self, M, N):
        global MAP, pil_image
        MAP = random_map(M, N, self.level)
        image = np.ones((M * W, N * W, 3), np.uint8) * 255
        for x in range(M):
            for y in range(N):
                if MAP[x][y] == '#':
                    image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_tuong
                else:
                    image[x * W:(x + 1) * W, y * W:(y + 1) * W] = mau_nen

        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.dem = 0
        self.path_found = False  # Đặt lại trạng thái tìm đường
        self.lbl_status.config(text="Chỉnh kích thước thành công!")

    def btn_mini_map_click(self):
        self.btn_reset_click()
        self.update_map_size(10, 30)
        self.size = 1

    def btn_medium_map_click(self):
        self.btn_reset_click()
        self.update_map_size(20, 60)
        self.size = 2

    def btn_large_map_click(self):
        self.btn_reset_click()
        self.update_map_size(40, 60)
        self.size = 3

    def btn_easy_level_click(self):
        self.level = 0.2
        self.lbl_status.config(text="Chọn mức thành công!")
        self.lbl_level.config(text="Mức đơn giản!")

    def btn_medium_level_click(self):
        self.level = 0.3
        self.lbl_status.config(text="Chọn mức thành công!")
        self.lbl_level.config(text="Mức bình thường!")

    def btn_hard_level_click(self):
        self.level = 0.5
        self.lbl_status.config(text="Chọn mức thành công!")
        self.lbl_level.config(text="Mức phức tạp!")

    def btn_random_map_click(self):
        self.update_map_image()
        self.image_tk = ImageTk.PhotoImage(pil_image)
        self.cvs_me_cung.create_image(0, 0, anchor=tk.NW, image=self.image_tk)
        self.dem = 0
        self.path_found = False  # Đặt lại trạng thái tìm đường
        self.lbl_status.config(text="Chỉnh map mới thành công!")

    def btn_find_path_click(self):
        # Kiểm tra xem đã chọn điểm bắt đầu và kết thúc chưa
        if 'o' not in [item for sublist in MAP for item in sublist] or 'x' not in [item for sublist in MAP for item in sublist]:
            self.lbl_status.config(text="Hãy chọn điểm!")
            return
        # Tạo một luồng để chạy việc tìm đường
        path_thread = threading.Thread(target=self.find_path)
        path_thread.start()

    def find_path(self):
        """Tìm đường trong mê cung và vẽ lên canvas"""
        self.lbl_status.config(text="Đang tìm đường...")
        problem = MazeSolver(MAP)
        result = astar(problem, graph_search=True)

        if result is None:
            self.lbl_status.config(text="Không tìm thấy đường đi!")
        else:
            path = [x[1] for x in result.path()]
            for i in range(1, len(path)- 1):
                x, y = path[i]
                # Vẽ đường đi màu xám
                self.cvs_me_cung.create_rectangle(x * W + 2, y * W + 2, (x + 1) * W - 2, (y + 1) * W - 2,
                                                  outline='#7f7f7f', fill='#7f7f7f')
                time.sleep(0.03)
                self.cvs_me_cung.update()
            self.lbl_status.config(text="Đã tìm thấy đường đi!")
            self.path_found = True  # Đặt trạng thái tìm thấy đường

if	__name__ ==	"__main__":
    app	=	App()
    app.mainloop()
