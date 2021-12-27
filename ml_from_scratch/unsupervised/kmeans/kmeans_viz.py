from kmeans import KMeansClustering
import sys
sys.path.insert(1,"src/animation")
# from general_data_anim_things import gen_utils
from manim import *
import csv
import numpy as np
# from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeansAnim(Scene):
    CONFIG = {
        "x_min": -5,
        "x_max": 5,
        "y_min": -5,
        "y_max": 5,
        "graph_origin": ORIGIN,
        "function_color": WHITE,
        "axes_color": WHITE
    }

    # CLUSTER_COLORS = [RED, GREEN, BLUE]

    def __init__(self,dir_to_data,**kwargs):
        self.coords = []
        self.load_data(dir_to_data)
        self.num_iter = 8
        Scene.__init__(self, **kwargs)
    def load_data(self, dir_to_data):
        with open(f'{dir_to_data}', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                x, y = row
                self.coords.append([float(x)/3, float(y)/3, 0])
        file.close()
    def gen_dots(self, t_stamp,colors=[RED,GREEN,BLUE]):
        ret = []
        self.colors=colors
        if colors != None:
            for coord, color in zip(self.coords, self.model.cluster_vals[t_stamp]):
                dot = Dot(coord[:2]+[0])
                dot.set_color(self.colors[color])
                dot.set_color(colors)
                ret.append(dot)
            return ret
        else:
            for coord in self.coords:
                dot = Dot(coord[:2]+[0])
                ret.append(dot)
            return ret


    def gen_centers(self, t_stamp):
        cents = []
        for center in self.model.center_vals[t_stamp]:
            point = Dot(list(center[:2]) + [0])
            cents.append(point)
        return cents

    def ret_centers_formatted(self, t_stamp):
        ret = []
        centers = self.model.center_vals[t_stamp]
        for center in centers:
            st = f"[{round(center[0], 2)}, {round(center[1], 2)}]"
            ret.append(st)
        return ret

    def gen_ctexts(self, t_stamp):
        ctext = self.ret_centers_formatted(t_stamp)
        center_text_0 = Tex(ctext[0], color=RED)
        center_text_0.scale(0.5)
        center_text_1 = Tex(ctext[1], color=GREEN)
        center_text_1.scale(0.5)
        center_text_2 = Tex(ctext[2], color=BLUE)
        center_text_2.scale(0.5)

        center_text_0.shift(2*DOWN, 6*LEFT)
        center_text_1.next_to(center_text_0, DOWN)
        center_text_2.next_to(center_text_1, DOWN)
        return center_text_0, center_text_1, center_text_2

    def disp_text(self, text, pos1, pos2, col1=WHITE, col2=None):
        ktitle = Tex(text)
        ktitle.shift(pos1, pos2)
        ktitle.scale(0.7)
        if not col2:
            col2 = col1
        ktitle.set_color_by_gradient(col1, col2)
        self.play(Write(ktitle))

    def construct(self):
        # self.setup_axes(animate=True)
        self.run_kmeans()

        title = Tex("K-Means Clustering:")
        title.set_color_by_gradient(BLUE, PURPLE)
        title.shift(3.5*UP)
        self.play(Write(title))

        # initial points
        old_dot_list = self.gen_dots(0)
        old_dots = VGroup(*old_dot_list)
        self.play(Create(old_dots))

        # initial centers:
        old_centers_list = self.gen_centers(0)
        old_centers = VGroup(*old_centers_list)
        self.play(Create(old_centers))

        self.disp_text("Centers:", 1.5*DOWN, 6*LEFT)
        self.disp_text("k=3", 1.5*DOWN, 6*RIGHT)
        self.disp_text("n=100", 2*DOWN, 6*RIGHT)

        old_c0, old_c1, old_c2 = self.gen_ctexts(0)
        self.play(Write(old_c0))
        self.play(Write(old_c1))
        self.play(Write(old_c2))

        old_vg = VGroup(old_dots, old_centers, old_c0, old_c1, old_c2)

        # transformation of the points
        for t in range(1, self.num_iter):
            dot_list = self.gen_dots(t)
            dots = VGroup(*dot_list)
            centers_list = self.gen_centers(t)
            centers = VGroup(*centers_list)
            c0, c1, c2 = self.gen_ctexts(t)
            vg = VGroup(dots, centers, c0, c1, c2)

            self.play(ReplacementTransform(old_vg, vg))
            old_vg = vg
            self.wait(0.5)


    def run_kmeans(self):
        self.model = KMeansClustering(3)
        # self.model=model
        print(np.array(self.coords)[:, :2])
        self.model.fit(np.array(self.coords)[:, :2], plot_final=False,
                       num_iter=self.num_iter, runs=1)


if __name__=="__main__":
    c=KMeansAnim("test_dat.csv")
    c.run_kmeans()