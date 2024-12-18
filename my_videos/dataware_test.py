from scipy.linalg import decomp
from manim_imports_ext import *
import time
class data_warehouse_part(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
    
        # kun svg
        # frame.to_default_state()

        # original_points=points[::2]
        # cloud=DotCloud(original_points,radius=0.02)
        # cloud.scale(2)
        # svg.scale(2)

        # mat
        # cloud_points=cloud.get_all_points().T[:2,:] 
        # def nparray_to_tex(arr):
        #     rows = []
        #     for row in arr:
        #         rows.append(" & ".join(f"{x:.2f}" for x in row))  
        #     latex_matrix = "\\left[\\enspace\\begin{array}{*{759}c}\n" + " \\\\\n".join(rows) + "\n\\end{array}\\enspace\\right]"
        #     return latex_matrix

        # mat=Matrix(np.round(cloud_points[:,:30],2))
        # swap_entries_for_ellipses_in_range(mat,10,19)
        # mat.to_corner(UL)
        # self.add(mat)
        # self.remove(mat)
        # frame.reorient(0, 0, 0, (18.37, 3.72, 0.0), 30.18)
        # self.play(FadeOutToPoint(mat.get_column(0).copy(),ax.c2p(*cloud_points[:,0])),
        	# ShowCreation(mob.pointwise_become_partial(cloud,0,1)))
        # self.play(ShowCreation(cloud))
        # self.add(cloud)

        # partial
        # cloudcopy=cloud.copy()
        # length=cloud_points.shape[1]
        # cloudcopy.pointwise_become_partial(cloud,0,300/length)
        # self.add(cloudcopy)
        # self.remove(cloudcopy)
        # func
        # def swap_entries_for_ellipses_in_range(matrix, start_col: int, end_col: int):
        #     rows = matrix.get_rows()
        #     cols = matrix.get_columns()


        #     if not (0 <= start_col < len(cols)) or not (0 <= end_col < len(cols)):
        #         raise ValueError("列索引超出范围")

        #     for col_index in range(start_col, end_col + 1):  # 迭代列索引
        #         matrix.swap_entries_for_ellipses(col_index=col_index)

        # three parts
        def get_cloud_grp(points,slices):
            grp=Group()
            for i in range(slices):
                cloud=DotCloud(points[i::slices,:],radius=0.05)
                grp.add(cloud)
            # grp.arrange(RIGHT)
            return grp
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        ax.add_coordinate_labels()
        ax.remove(ax.z_axis)
        ax.set_opacity(0.7)
        svg=SVGMobject('kun.svg',stroke_color=WHITE,stroke_opacity=1,stroke_width=1,)
        svg.scale(2)
        points=svg.get_all_points()             # 1518
        clouds=get_cloud_grp(points,slices=66)  # 1518=23*66
        # clouds.scale(2)
        cloud_0_points=np.round(clouds[0].get_all_points().T,2)[:2,:]
        mat=Matrix(cloud_0_points)

        # matrix_init
        mat.to_corner(UL)
        mat.set_opacity(1)
        mat.clear_updaters()
        self.add(mat)
        self.add(ax)
        # self.play(mat.animate.to_corner(UR),run_time=5,rate_func=there_and_back)
        # def grow_dots_animation(i):

        # animations
        the_points_grp=Group()
        def grow_dots_anim(cloud,i,run_time=1):
            the_point=cloud.copy().pointwise_become_partial(clouds[0],0,(i+1)/num_of_points)
            the_col=mat.get_column(i)
            the_coord=cloud.get_all_points()[i,:]
            arrow=Arrow(start=ORIGIN,end=the_coord,buff=0)
            mat.get_column(i).set_opacity(0.5)
            self.play(TransformFromCopy(the_col,arrow),run_time=run_time/2)
            self.play(FadeOutToPoint(arrow,the_coord),FadeIn(the_point),run_time=run_time/2)
            the_points_grp.add(the_point)

        # compute whole_time
        num_of_points=cloud_0_points.shape[1]
        whole_time=0
        def func(i):
            return 2/math.sqrt(i)
        for i in np.arange(5,num_of_points):
            run_time=func(i)
            whole_time+=run_time
        print(f"whole time is {whole_time}")

        # anims
        velocity=(mat.get_right()[0]-frame.get_shape()[0]/2)/(whole_time+1.8*3)
        grow_dots_anim(clouds[0],0,run_time=2)
        grow_dots_anim(clouds[0],1,run_time=2)
        mat.add_updater(lambda m,dt:m.shift(dt*LEFT*velocity))
        grow_dots_anim(clouds[0],2,run_time=1.8)
        grow_dots_anim(clouds[0],3,run_time=1.8)
        grow_dots_anim(clouds[0],4,run_time=1.8)

        for i in np.arange(5,num_of_points):
            run_time=func(i)
            grow_dots_anim(clouds[0],i,run_time=run_time)
        mat.clear_updaters()

        # many many points
        left_bracket=Tex(R"[")
        number=Tex('2',font_size=48)
        times=Tex(R'\times')
        decimal_number=DecimalNumber(23,num_decimal_places=0,color=TEAL)
        decimal_number.scale(0.7)
        right_bracket=Tex(R"]")
        grp=VGroup(left_bracket,number,times,decimal_number,right_bracket).arrange(RIGHT)
        grp.match_height(mat)
        grp.to_edge(UP)
        # self.add(grp)
        self.play(FadeTransform(mat.brackets[0],left_bracket),
            FadeTransform(mat.brackets[1],right_bracket),
            ReplacementTransform(VGroup(mat.elements),VGroup(number,times,decimal_number)),
            run_time=2)

        # shuffle clouds
        clouds_shuffled=clouds[1:].shuffle()

        def add_cloud(i,clouds,number,value):
            value=value+value*(i+1)
            self.play( number.animate.set_value(value),
                ShowCreation(clouds_shuffled[i]) ,run_time=1)


        for i in np.arange(0,10):
            add_cloud(i,clouds_shuffled,decimal_number,num_of_points)

        self.play(LaggedStartMap(ShowCreation,clouds_shuffled[10:],run_time=8),
            decimal_number.animate.set_value(len(points)).set_anim_args(run_time=2),
            right_bracket.animate.shift(RIGHT).set_anim_args(run_time=2))

        # fadeout drawborder
        self.play(LaggedStartMap(FadeOut,grp,shift=UP),FadeOut(ax))
        self.play(DrawBorderThenFill(svg,run_time=10,stroke_width=3),
            LaggedStartMap(FadeOut,Group(clouds_shuffled,the_points_grp),run_time=10))

        # back rectangle
        rec=BackgroundRectangle(svg,color=WHITE,buff=0,fill_opacity=1)
        svg.set_fill(color=BLACK,opacity=1)
        rec.next_to(svg,UP)

        # image
        im=ImageMobject('kun.jpg')
        im.rescale_to_fit(rec.get_width(),0)
        im.set_opacity(1)

        # rec animation
        rec.stretch_to_fit_height(im.get_height())
        self.play(FadeIn(rec),run_time=0.3)
        self.bring_to_back(rec)
        self.play(rec.animate.move_to(svg,aligned_edge=DOWN),run_time=3)


        # image anim
        im.move_to(rec)
        self.play(FadeTransform(Group(rec,svg),im),run_time=2,rate_func=smooth)

        # self.add(im)

        # mat.clear_updaters()
        # swap_entries_for_ellipses_in_range(mat,0,22)
        # mat.add_updater(lambda m,dt:m.shift(-dt*LEFT*velocity))
        # self.wait(5)

        # dt updater
        # dot=Dot().set_color(RED)
        # mat1=Matrix([[1,2],[3,4]])
        # self.add(mat1)
        # mat1.add_updater(lambda m,dt:m.shift(dt*LEFT))
        # self.wait(1)
        # mat2=mat1.deepcopy()
        # mat2.get_column(0).set_opacity(0.5)
        # self.play(mat1.animate.become(mat2))

        # self.wait(1)
        # the_point=clouds[0].copy().pointwise_become_partial(clouds[0],0,2/num_of_points)
        # the_col=mat.get_column(1)
        # the_coord=clouds[0].get_all_points()[1,:]

        # arrow=Arrow(start=ORIGIN,end=the_coord,buff=0)
        # self.play(TransformFromCopy(the_col,arrow),the_col.animate.set_opacity(0.5))
        # self.play(FadeOutToPoint(arrow,the_coord),FadeIn(the_point))
        # shuffle animations
        # self.play(LaggedStartMap(ShowCreation,clouds[1:].shuffle()),run_time=10)
        








class rotation(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame

        # rotation
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        im=ImageMobject('kun.jpg')
        surface_base=Sphere(radius=5,resolution=(101,101))
        texture=TexturedSurface(surface_base,'basketball.jpg')
        arrow_x=Arrow(RIGHT*4.8,RIGHT*8,thickness=20,fill_color=GREEN)
        arrow_y=Arrow(UP*4.8,UP*8,thickness=20,fill_color=RED)
        arrow_z=Arrow(OUT*4.8,OUT*8,thickness=20,fill_color=BLUE)
        arrow_z.rotate(PI/2,axis=OUT)
        arrows=VGroup(arrow_x,arrow_y,arrow_z)
        arrows.apply_depth_test()
        frame.reorient(29, 41, 0, (0.41, 1.06, 0.56), 14.22)
        tex_x=Tex('x',font_size=300)
        tex_x.next_to(arrow_x,RIGHT).match_color(arrow_x)
        tex_y=Tex('y',font_size=300)
        tex_y.next_to(arrow_y,UP).match_color(arrow_y)
        tex_z=Tex('z',font_size=300)
        tex_z.next_to(arrow_z,OUT,buff=1).match_color(arrow_z)
        tex_z.rotate(PI/2,axis=RIGHT)
        texs=VGroup(tex_x,tex_y,tex_z)
        basketball=Group(texture,arrows,texs)

        # matrix
        rotmat=Tex(R"""\left[\enspace\begin{matrix}
                    \cos \theta & -\sin \theta \\[1.2mm]
                    \sin \theta & \cos \theta
                    \end{matrix}\enspace
                    \right]""",
                    t2c={R"\theta":YELLOW})
        rotmat.to_corner(UL,buff=0.5)
        def get_rotmat(theta):
            degrees=theta*DEGREES
            rot_mat=np.array([[np.cos(degrees),-np.sin(degrees)],
                            [np.sin(degrees),np.cos(degrees)]])
            m=Tex(Rf"""\left[\enspace\begin{{matrix}}
                    {round(np.cos(degrees),2)} & {-round(np.sin(degrees),2)} \\[1.2mm]
                    {round(np.sin(degrees),2)} & {round(np.cos(degrees),2)} 
                    \end{{matrix}}\enspace
                    \right]""")
            # m=Matrix(rot_mat)
            m.to_corner(UR)
            return m
        rotmat2=get_rotmat(30).match_height(rotmat)
        self.add(rotmat)
        self.add(rotmat2)

        arrow0=Arrow(rotmat.get_right(),rotmat2.get_left())
        self.add(arrow0)
        tex_theta=Tex(R"""\theta=30^\circ""")
        tex_theta.next_to(arrow0,UP)
        self.add(tex_theta)
        self.add(basketball)
        self.add(im)
        # scale
        self.add(ax)
        frame.to_default_state()
        im.apply_depth_test()
        basketball.center()
        basketball.shift(LEFT)
        basketball.shift(DOWN)
        basketball.shift(IN*0.5)
        basketball.scale(0.15)
        texture.center()
        # basketball.rotate(70*DEGREES,axis=OUT)
        # basketball.center()


        pass