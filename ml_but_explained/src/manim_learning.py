# from manimlib.imports import *
# from manim import Circle,Square,Transform,Create,FadeOut,PINK,PI,Scene,Dot
from manim import *
from sklearn.preprocessing import StandardScaler



# from manim.imports import Scene
class Shapes(Scene):
    def mins_and_maxes(self,dataset):
        xvals=[i[0] for i in dataset]
        yvals=[i[1] for i in dataset]
        self.xmin=min(xvals)-1
        self.xmax=max(xvals)+1
        self.ymin=min(yvals)-1
        self.ymax=max(yvals)+1

    def construct(self):
        dataset = [[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9,11]]
        self.mins_and_maxes(dataset)
        ax=Axes(x_range=[self.xmin,self.xmax,1],y_range=[self.ymin,self.ymax,1]).add_coordinates()
        for i in dataset:
            c=Dot(ax.coords_to_point(i[0],i[1]))
            self.add(c)
        plane=NumberPlane()
        self.add(ax)


