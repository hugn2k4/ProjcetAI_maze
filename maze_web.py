import streamlit as st
import numpy as np
import cv2
from PIL import Image
import random
import math
from simpleai.search import SearchProblem, astar

# Định nghĩa các biến và cấu hình cho mê cung
cost_regular = 1.0
cost_diagonal = 1.7

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


# Hàm tạo bản đồ ngẫu nhiên với mật độ tường nhất định
def random_map(height, width, wall_density):
    # Tạo ma trận toàn đường đi (trống)
    maze = [[' ' for _ in range(width)] for _ in range(height)]

    # Tạo viền tường xung quanh
    for i in range(width):
        maze[0][i] = '#'
        maze[height - 1][i] = '#'
    for i in range(height):
        maze[i][0] = '#'
        maze[i][width - 1] = '#'

    # Hàm kiểm tra ô hợp lệ
    def is_valid_move(x, y):
        return 1 <= x < height - 1 and 1 <= y < width - 1 and maze[x][y] == ' '

    # Đệ quy tạo tường
    def carve_path(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # Bước nhảy
        random.shuffle(directions)  # Xáo trộn để tạo tính ngẫu nhiên
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            wall_x, wall_y = x + dx // 2, y + dy // 2
            if is_valid_move(nx, ny):
                # Đặt tường tại ô giữa
                maze[wall_x][wall_y] = '#'
                # Đặt tường tại ô tiếp theo
                maze[nx][ny] = '#'
                # Tiếp tục đệ quy từ ô tiếp theo
                carve_path(nx, ny)

    # Bắt đầu từ ô trung tâm
    carve_path(height // 2, width // 2)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if maze[y][x] == '#' and random.random() > wall_density:
                maze[y][x] = ' '  # Xóa tường trong hàng

    return maze


# Hiển thị mê cung dưới dạng ảnh
def display_maze(maze, size,is_path=False):
    W = 50  # Đặt kích thước của từng ô trong mê cung
    M, N = len(maze), len(maze[0])
    image = np.ones((M * W, N * W, 3), np.uint8) * 255

    mau_tuong = (82, 110, 72)  # Màu tường
    mau_nen = (194, 255, 199)  # Màu nền

    for y in range(M):
        for x in range(N):
            color = mau_tuong if maze[y][x] == "#" else mau_nen
            image[y * W:(y + 1) * W, x * W:(x + 1) * W] = color

            # Hiển thị các điểm đặc biệt (bắt đầu, kết thúc, đường đi)
            if maze[y][x] == 'o':  # Điểm bắt đầu
                cv2.circle(image, (x * W + W // 2, y * W + W // 2), W // 4, (0, 255, 0), -1)
            elif maze[y][x] == 'x':  # Điểm kết thúc
                cv2.circle(image, (x * W + W // 2, y * W + W // 2), W // 4, (255, 0, 0), -1)

            # Nếu is_path là True, vẽ đường đi
            if is_path and maze[y][x] == '*':  # Đường đi
                cv2.rectangle(image, (x * W, y * W), ((x + 1) * W, (y + 1) * W), (169, 169, 169), thickness=-1)
                cv2.circle(image, (x * W + W // 2, y * W + W // 2), W // 4, (255, 0, 255), -1)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Class giải quyết bài toán mê cung
class MazeSolver(SearchProblem):
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

    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#":
                actions.append(action)
        return actions

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
        return (x, y)

    def is_goal(self, state):
        return state == self.goal

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

# Hàm trang chủ
def home_page():
    st.title("Báo cáo đồ án cuối kì")
    st.image("LogoHCMUTE.png", width=200)
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("Thông tin cá nhân")
    st.write("Môn học: **Trí tuệ nhân tạo**")
    st.write("Giảng viên hướng dẫn: **Trần Tiến Đức**")
    st.write("Họ và tên: **Lê Công Hùng**")
    st.write("MSSV: **22110151**")
    st.markdown(
        """
        <style>
        .custom-text {
            font-size: 18px;
            color: #ff5733;
            font-weight: bold;
        }
        </style>
        <div class="custom-text">
            Đây là đồ án cuối kì môn Trí tuệ nhân tạo, với bài toán giải mê cung sử dụng thuật toán A*.
        </div>
        """, unsafe_allow_html=True
    )
# Hàm trang Mê cung
def maze_page():
    # Cấu hình Streamlit cho ứng dụng giải mê cung
    st.title("Giải Mê Cung")
    st.sidebar.header("Cài đặt")

    def on_sidebar_change():
        # Xóa tất cả các giá trị trong session_state khi thay đổi sidebar
        st.session_state.clear()

    # Kích thước và độ phức tạp của bản đồ
    map_size = st.sidebar.selectbox("Chọn kích thước bản đồ", ["10x10", "20x20", "40x40"], index=0,
                                    on_change=on_sidebar_change)
    wall_density = st.sidebar.slider("Mức độ phức tạp của tường", 0.1, 1.0, 0.3, on_change=on_sidebar_change)

    height, width = map(int, map_size.split("x"))

    if "maze" not in st.session_state:
        st.session_state["maze"] = random_map(height, width, wall_density)
    # Hiển thị mê cung

    maze = st.session_state["maze"]
    maze[1][1] = 'o'
    maze[int(map_size.split('x')[1]) - 2][int(map_size.split('x')[0]) - 2] = 'x'
    maze_image = display_maze(maze, map_size)
    st.image(maze_image, caption="Bản đồ mê cung", use_container_width=True)

    st.sidebar.header("Chức năng")
    # Bắt đầu giải mê cung
    if st.sidebar.button("Tìm đường đi"):
        problem = MazeSolver(maze)
        result = astar(problem, graph_search=True)
        if result:
            path = [x[1] for x in result.path()]
            for (x, y) in path[1:-1]:
                maze[y][x] = "*"
            # Hiển thị mê cung đã có đường đi
            solved_image = display_maze(maze, map_size, True)
            st.image(solved_image, caption="Đường đi trong mê cung", use_container_width=True)
        else:
            st.error("Không tìm thấy đường đi!")

    if st.sidebar.button("Làm mới lại"):
        st.empty()

    if st.sidebar.button("Bản đồ ngẫu nhiên"):
        st.session_state["maze"] = random_map(height, width, wall_density)
        st.sidebar.success("Bản đồ ngẫu nhiên đã được tạo! Nhấn 'Làm mới lại' để tải lại bản đồ.")
        st.empty()


# Giao diện sidebar để chọn trang
st.sidebar.title("Project cuối kì")
page = st.sidebar.radio("Menu:", ("Trang Chủ", "Mê Cung"))

# Hiển thị nội dung trang dựa trên lựa chọn
if page == "Trang Chủ":
    home_page()
elif page == "Mê Cung":
    maze_page()