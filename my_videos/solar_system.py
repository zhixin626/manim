from manimlib import *
import re
import numpy as np
def get_orthogonal_proj_matrix(A):
    proj_mat=np.dot(np.dot(A,np.linalg.inv(np.dot(A.T,A))),A.T)
    return proj_mat

class rotation(InteractiveScene):
    def construct(self):
        # start
        # Scene()
        # CameraFrame()
        # Mobject()
        
        # add universe
        frame=self.frame
        frame.get_implied_camera_location()
        sf2=Sphere(radius=100,color=GREY,opacity=0)
        t_sf2=TexturedSurface(sf2,"./assets/milkyway.jpg")
        t_sf2.set_shading(0.3,0.1,0.6)
        # self.add(t_sf2)
        frame.reorient(84, 87, 0, (4.04, 1.68, 0.46), 12.85)
        self.play(FadeIn(t_sf2))

        # orthogonal projection:
        ax=ThreeDAxes()
        # ax.apply_depth_test()
        ax.add_axis_labels(font_size=80)
        A=np.array([[1,0],[0,1],[0,0]])
        mat=get_orthogonal_proj_matrix(A)
        sf=Sphere()
        t_sf=TexturedSurface(sf,"./assets/day.jpg","./assets/night.jpg")
        t_sf.add(SurfaceMesh(sf).set_stroke(BLUE,1,opacity=0.3))
        t_sf.move_to(np.array([3,0,0]))
        # self.frame.reorient(1, 66, 0, (1.68, 0.16, 0.25), 4.99)
        # self.add(ax)
        # self.add(t_sf)
        self.play(ShowCreation(ax))
        self.play(frame.animate.reorient(11, 67, 0, (1.29, 0.26, 0.5), 9.38),
            run_time=2)
        self.play(FadeIn(t_sf))

        # add light
        light = self.camera.light_source
        light_dot = GlowDot(color=WHITE, radius=0.3)
        light_dot.always.move_to(light)
        light.move_to(ORIGIN)
        self.add(light_dot)

class solar_system(InteractiveScene):
    def construct(self):
        # start
        frame=self.frame
        ax=ThreeDAxes((-200,200),(-200,200),(-200,200))
        sun=Sphere(radius=109)
        mercury=Sphere(radius=0.38,resolution=(51,101))
        venus=Sphere(radius=0.94,resolution=(70,25))
        earth=Sphere(radius=1,resolution=(101,101))
        moon=Sphere(radius=0.27,resolution=(101,51))           # radius of earth = 1
        mars=Sphere(radius=0.53,resolution=(51,101))
        jupiter=Sphere(radius=11,resolution=(51,101))
        saturn=Sphere(radius=9,resolution=(51,101))
        uranus=Sphere(radius=4,resolution=(51,101))
        neptune=Sphere(radius=3.8,resolution=(51,101))

        factor=23454     # 1 AU = 23454 * radius of earth
        correction_vector=np.array([factor,0,0])
        sun.move_to(np.array([0,0,0])-correction_vector)
        mercury.move_to(np.array([factor*0.38,0,0])-correction_vector)
        venus.move_to(np.array([factor*0.72,0,0])-correction_vector)
        earth.move_to(np.array([factor*1,0,0])-correction_vector)
        moon.move_to(np.array([factor*1.00257,0,0])-correction_vector)
        mars.move_to(np.array([factor*1.52,0,0])-correction_vector)
        jupiter.move_to(np.array([factor*5.20,0,0])-correction_vector)
        saturn.move_to(np.array([factor*9.58,0,0])-correction_vector)
        uranus.move_to(np.array([factor*19.14,0,0])-correction_vector)
        neptune.move_to(np.array([factor*30.2,0,0])-correction_vector)

        # universe background
        universe=Sphere(radius=factor*100)
        t_universe=TexturedSurface(universe,"milkyway.jpg").set_shading(0.3,0.1,0.6)

        # textured planet
        t_sun=TexturedSurface(sun,"sun.jpg")
        t_mercury=TexturedSurface(mercury,"mercury.jpg")
        t_venus=TexturedSurface(venus,"venus.jpg")
        t_earth=TexturedSurface(earth,"day.jpg","night.jpg")
        t_moon=TexturedSurface(moon,"moon.jpg")
        t_mars=TexturedSurface(mars,"mars.jpg")
        t_jupiter=TexturedSurface(jupiter,"jupiter.jpg")
        t_saturn=TexturedSurface(saturn,"saturn.jpg")
        t_uranus=TexturedSurface(uranus,"uranus.jpg")
        t_neptune=TexturedSurface(neptune,"neptune.jpg")

        # light
        light=self.camera.light_source
        light.move_to(sun.get_center())

        # set shading
        t_sun.set_shading(0.3,0.2,0.1)
        t_earth.set_shading(0.2,0.3,0)
        t_venus.set_shading(0.1,0.1,0.2)
        t_mercury.set_shading(0.05,0.2,0.2)
        t_mars.set_shading(0.05,0.1,0.2)
        t_jupiter.set_shading(0.05,0.1,0.1)
        t_saturn.set_shading(0,0.4,0.2)
        t_uranus.set_shading(0,0.1,0.1)
        t_neptune.set_shading(0,0.1,0.1)

        # mesh
        earth_mesh=SurfaceMesh(earth)

        # self.add(t_universe,sun,mercury,venus,earth,mars,jupiter,saturn,uranus,neptune)

        self.add(t_universe,t_sun,t_mercury,t_venus,t_earth,t_moon,
            t_mars,t_jupiter,t_saturn,t_uranus,t_neptune)
        frame.reorient(-4, 66, 0, sun.get_center(), 503.37)
        frame.reorient(-4, 66, 0, mercury.get_center(), 3)
        frame.reorient(-4, 66, 0, venus.get_center(), 3)
        frame.reorient(-4, 66, 0, earth.get_center(), 3)
        frame.reorient(83, 78, 0, moon.get_center()+np.array([0,moon.radius,0]), 0.1)  # moon
        frame.reorient(-4, 66, 0, mars.get_center(), 3)
        frame.reorient(-4, 66, 0, jupiter.get_center(), 33)
        frame.reorient(-4, 66, 0, saturn.get_center(), 25)
        frame.reorient(-4, 66, 0 ,uranus.get_center(), 10)
        frame.reorient(-4, 66, 0, neptune.get_center(), 10)

        # from Earth back to Sun
        frame.reorient(4, 73, 0, earth.get_center(),3.00)
        self.play(frame.animate.reorient(68, 82, 0, earth.get_center(), 3))
        self.play(frame.animate.reorient(68, 82, 0, (-0.59, -1.29, 0.47), 0.16),
            run_time=3)
        self.play(frame.animate.reorient(68, 82, 0, venus.get_center(), 3),
            run_time=3,rate_func=smooth)
        self.play(frame.animate.reorient(68, 82, 0, mercury.get_center(), 3),
            run_time=5,rate_func=smooth)
        self.play(frame.animate.reorient(68, 82, 0, sun.get_center(), 500),
            run_time=5,rate_func=smooth)

        # orthogonal projection matrix
        mat=np.array([[1,0,0],[0,1,0],[0,0,0.01]])
        self.play(t_earth.animate.apply_matrix(mat))      

class test_sphere_resolution(InteractiveScene):
    def construct(self):
        # start
        frame=self.frame
        factor=23455
        correction_vector=np.array([factor,0,0])

        mercury=Sphere(radius=0.38,resolution=(52,101))
        mercury.move_to(np.array([factor*0.38,0,0])-correction_vector)
        t_mercury=TexturedSurface(mercury,"mercury.jpg","mercury_night.jpg")
        self.add(t_mercury)
        frame.reorient(-6,68,0,mercury.get_center(),1)

class Text2(VGroup):
    def __init__(self, 
        english,chinese,direction=DOWN,
        font1="Mongolian Baiti",font2="SJsuqian",
        font_size1=48,font_size2=40,
        buff=0.2,
         **kwargs):
        text_A = Text(english, font=font1, font_size=font_size1,**kwargs)
        text_B = Text(chinese, font=font2,font_size=font_size2, **kwargs)
        super().__init__(text_A, text_B)
        self.en=text_A
        self.ch=text_B
        text_B.next_to(text_A, direction,buff=buff)
class Matrix_usefulness(InteractiveScene):
    def construct(self):
        frame=self.frame
        # start
        # equation part
        tex_mat=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1&2&3\\
            4&5&6\\
            7&8&9
            \end{matrix}
            \enspace\right]
                    """,
                    font_size=60,
                    )
        tex_mat_aug=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1&2&3\\
            4&5&6\\
            7&8&9
            \end{matrix}
            \enspace\right.
            \left|\enspace
            \begin{matrix}6\\15\\24
            \end{matrix}
            \enspace\right]
            """,font_size=60)
        tex_eqn=TexText(R"""
            \begin{alignat*}{6}
             1\,&x+&&\,2\,&&y\,+&&\,3\,&&z&&=6  \\
            4\,&x+&&\,5\,&&y\,+&&\,6\,&&z&&=15\\
            7\,&x+&&\,8\,&&y\,+&&\,9\,&&z&&=24
            \end{alignat*}
            """,
            t2c={"x":BLUE_A,"y":BLUE_B,"z":BLUE_C},
            isolate=["+"]
                        )
        grp=VGroup(tex_mat_aug,tex_eqn).arrange(RIGHT,buff=1)
        text1=Text2('Matrix','矩阵',DOWN,buff=0.4).next_to(tex_mat_aug,UP)
        text2=Text2('Augmented Matrix','增广矩阵',DOWN,buff=0.2).next_to(tex_mat_aug,UP)
        saved_state=tex_mat_aug[22:24].copy()           # bracket original position
        tex_mat_aug[22:24].move_to(tex_mat_aug[11:17])  # move bracket
    
        # animation for equations
        kw=dict(path_arc=30*DEGREES,run_time=1)
        tex_eqn.save_state()
        self.play(Write(tex_eqn.center().scale(1.3)))
        self.play(tex_eqn.animate.restore())
        self.play(Write(tex_mat_aug[22:24]),Write(tex_mat_aug[0:2]))
        Animations=[
        TransformFromCopy(VGroup(tex_eqn[0],tex_eqn[10],tex_eqn[21]),
          VGroup(tex_mat_aug[2],tex_mat_aug[5],tex_mat_aug[8]),**kw),
        VGroup(tex_eqn[0],tex_eqn[10],tex_eqn[21]).animate.set_opacity(0.5),
        TransformFromCopy(VGroup(tex_eqn[3],tex_eqn[13],tex_eqn[24]),
          VGroup(tex_mat_aug[3],tex_mat_aug[6],tex_mat_aug[9]), **kw),
        VGroup(tex_eqn[3],tex_eqn[13],tex_eqn[24]).animate.set_opacity(0.5),
        TransformFromCopy(VGroup(tex_eqn[6],tex_eqn[16],tex_eqn[27]),
          VGroup(tex_mat_aug[4],tex_mat_aug[7],tex_mat_aug[10]), **kw),
        VGroup(tex_eqn[6],tex_eqn[16],tex_eqn[27]).animate.set_opacity(0.5),
                   ]
        self.play(LaggedStart(Animations,lag_ratio=0.1))
        self.play(Write(text1))
        self.play(tex_mat_aug[22:24].animate.move_to(saved_state)
            ,FadeIn(tex_mat_aug[11:17]))
        self.play(
            TransformFromCopy(VGroup(tex_eqn[9],tex_eqn[19:21],tex_eqn[30:32]),
          VGroup(tex_mat_aug[17],tex_mat_aug[18:20],tex_mat_aug[20:22]), **kw),
            VGroup(tex_eqn[9],tex_eqn[19:21],tex_eqn[30:32]).animate.set_opacity(0.5))
        self.play(
            ReplacementTransform(text1.en.get_part_by_text('Matrix'),
                text2.en.get_part_by_text('Matrix')),
            ReplacementTransform(text1.ch.get_part_by_text('矩阵'),
                text2.ch.get_part_by_text('矩阵')),
            Write(text2.en.get_part_by_text('Augmented')),
            Write(text2.ch.get_part_by_text('增广'))
            )
        self.play(FadeOut(tex_eqn),VGroup(tex_mat_aug,text2).animate.center())
        self.play(Uncreate(tex_mat_aug),Uncreate(text2.en),Uncreate(text2.ch))
        # text_mat_nd_initialize
        tex_mat_1d=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1\\
            \end{matrix}
            \enspace\right]
                    """, font_size=60,)
        tex_mat_2d=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1&0\\[0.5mm]
            0&1
            \end{matrix}
            \enspace\right]
                    """, font_size=60,)
        tex_mat_3d=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1&0&0\\[0.5mm]
            0&1&0\\[0.5mm]
            0&0&1
            \end{matrix}
            \enspace\right]
                    """, font_size=60,)
        tex_mat_4d=Tex(R"""
            \left[\enspace
            \begin{matrix}
            1&0&0&0\\[0.5mm]
            0&1&0&0\\[0.5mm]
            0&0&1&0\\[0.5mm]
            0&0&0&1
            \end{matrix}
            \enspace\right]
                    """, font_size=60,)
        grp_mats=VGroup(tex_mat_1d,tex_mat_2d,tex_mat_3d,tex_mat_4d).arrange(RIGHT)
        # animations
        tex_mat_1d.to_corner(UL)
        nbl=NumberLine((-10,10))
        nbl.add_numbers()
        a=Tex(R'\mathrm{a}')
        a_vec=tex_mat_1d.copy()
        a_number=DecimalNumber(0,num_decimal_places=1,
                                include_sign=True,
                                font_size=38)
        a_grp=VGroup(a,a_vec).arrange(RIGHT).to_corner(UR)
        a_number.next_to(a_vec,LEFT)
        dot=GlowDot(radius=0.5).move_to(nbl.n2p(a_number.get_value()))
        self.play(Write(tex_mat_1d))
        # self.add(a,a_vec,dot)
        self.play(TransformFromCopy(tex_mat_1d,a_vec),Write(a),run_time=2)
        self.play(FadeIn(dot),Write(nbl.numbers[abs(nbl.x_min)]),
            ReplacementTransform(a[0],a_number))
        a_number.add_updater(
            lambda m:m.set_value(nbl.p2n(dot.get_center()))
            )

        # state1
        line=Line(start=nbl.get_start(),end=nbl.get_end())
        a_number=DecimalNumber(0,num_decimal_places=1,
                                include_sign=True,
                                font_size=38).to_corner(UR)
        a_number.add_updater(lambda m:m.set_value(nbl.p2n(dot.get_center())))

        self.play(dot.animate.move_to(nbl.n2p(nbl.x_max)),
            ShowCreation(line.get_subcurve(0.5,1)),
            Write(nbl.numbers[abs(nbl.x_min)+1:]),
            frame.animate.reorient(0, 0, 0, (0,0,0), nbl.x_max+1.2),
            run_time=2)
        self.play(dot.animate.move_to(nbl.n2p(0)),run_time=2,rate_func=linear)
        grp=nbl.numbers[0:abs(nbl.x_min)]
        new_grp=VGroup(VMobject(),VMobject())
        for item in grp:
            new_grp.add_to_back(item)
        self.play(dot.animate.move_to(nbl.n2p(nbl.x_min)),
            ShowCreation(line.get_subcurve(0.5,0)),
            Write(new_grp,lag_ratio=1),run_time=2,rate_func=linear)
        self.play(dot.animate.move_to(nbl.n2p(0)),
            frame.animate.to_default_state(),run_time=1.5)
        self.clear()

        # matrix_arr_Setup
        mat1_arr=np.array([[1]])
        mat2_arr=np.array([[1,0],[0,1]])
        mat3_arr=np.array([[1,-2,3],[1,1,-3],[1,1,2]])
        mat4_arr=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        # matrix animations1
        mat3=MatrixCustom(mat3_arr)  
        linear_comb3=mat3.get_linear_combination()
        mat3.fix_in_frame()
        linear_comb3.fix_in_frame() 
        self.play(Write(mat3))
        Animations1=[
            AnimationGroup([TransformFromCopy(mat3.columns[0],VGroup(*linear_comb3[1].elements)),
                            TransformFromCopy(mat3.brackets,linear_comb3[1].brackets)]),
            AnimationGroup([TransformFromCopy(mat3.columns[1],VGroup(*linear_comb3[4].elements)),
                            TransformFromCopy(mat3.brackets,linear_comb3[4].brackets)]),
            AnimationGroup([TransformFromCopy(mat3.columns[2],VGroup(*linear_comb3[7].elements)),
                            TransformFromCopy(mat3.brackets,linear_comb3[7].brackets)])
            ]
        self.play(LaggedStart(Animations1,lag_ratio=0.8))
        Animations2=[
            Write(VGroup(linear_comb3[0],linear_comb3[3],linear_comb3[6])),
            GrowFromCenter(linear_comb3[2]),GrowFromCenter(linear_comb3[5])
        ]
        self.play(LaggedStart(Animations2,lag_ratio=0.5))
        # show axes animations
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        ax.add_coordinate_labels()
        self.play(Write(ax),run_time=2)
        self.play(frame.animate.reorient(18, 47, 0, (0.76, 1.97, 1.03), 6.88))


        # column vector to coordinates
        def get_lines(axis,one_dim_matrix):
            if isinstance(one_dim_matrix,Matrix):
                if not hasattr(one_dim_matrix, 'nparr'):
                    raise AttributeError("vector has no nparr")
                vector=one_dim_matrix.nparr
            vector=vector.flatten()
            xy_point=vector.copy()
            xy_point[2]=0
            line1=axis.get_line_from_axis_to_point(0,xy_point)
            line2=axis.get_line_from_axis_to_point(1,xy_point)
            line3=axis.get_line_from_axis_to_point(2,xy_point)
            line4=DashedLine(axis.c2p(*xy_point),axis.c2p(*vector))
            line5=axis.get_line_from_axis_to_point(2,axis.c2p(*vector))
            arrow=Arrow(axis.c2p(0,0,0),axis.c2p(*vector),buff=0,thickness=2)
            arrow.match_color(one_dim_matrix)
            unit_normal=arrow.get_unit_normal()
            angle=angle_between_vectors(unit_normal,cross(OUT,arrow.get_end()))
            arrow.rotate(angle,arrow.get_vector())
            grp=VGroup(VGroup(line1,line2),VGroup(line3,line4),line5)
            return grp,arrow
        def get_xyz(camera_position,z1=OUT):
            # x_vector=camera_position
            # y_vector=in the plane (z_axis and x_vector)
            # z_vector=cross product of x_vector and y_vector
            x=camera_position
            y=np.cross(np.cross(x,z1),x)
            z=np.cross(x,y)
            return normalize(x),normalize(y),normalize(z)
        def get_rotation_matrix(camera_postion): # combine new basis
            x,y,z=get_xyz(camera_postion)
            B=np.array([x,y,z]).T
            return B

        class TransformFromCopy2(Transform):
            def __init__(self, mobject, target_mobject, **kwargs):
                super().__init__(mobject.copy(), target_mobject.copy(), **kwargs)
            def clean_up_from_scene(self, scene: Scene) -> None:
                scene.remove(self.mobject)
                scene.remove(self.target_mobject)
        def get_animations(linear_comb_mat:Matrix,ThreeDAxes) -> list:
            # linear_comb_mat must have nparr attribute
            ax=ThreeDAxes
            nparr=linear_comb_mat.nparr.flatten()
            x_axis_correct_num=-ax.x_axis.x_min
            y_axis_correct_num=-ax.y_axis.x_min
            z_axis_correct_num=-ax.z_axis.x_min
            a=ax.x_axis.numbers[int(nparr[0]+x_axis_correct_num)]
            b=ax.y_axis.numbers[int(nparr[1]+y_axis_correct_num)]
            c=ax.z_axis.numbers[int(nparr[2]+z_axis_correct_num)]
            Animations=[
              TransformFromCopy2(linear_comb_mat.elements[0],a),
              TransformFromCopy2(linear_comb_mat.elements[1],b),
              TransformFromCopy2(linear_comb_mat.elements[2],c),]
            return Animations
        # GrowArrowAnimations
        self.play(frame.animate.reorient(20, 55, 0, (0.82, 1.28, 0.64), 4.69))
        Animations=get_animations(linear_comb3[1],ax)
        self.play(LaggedStart(Animations),run_time=2)
        grp,arrow=get_lines(ax,linear_comb3[1])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow))
        self.play(FadeOut(grp))

        self.play(frame.animate.reorient(-35, 59, 0, (-0.22, 0.08, 0.91), 4.69))
        Animations=get_animations(linear_comb3[4],ax)
        self.play(LaggedStart(Animations),run_time=2)
        grp,arrow=get_lines(ax,linear_comb3[4])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow))
        self.play(FadeOut(grp))

        self.play(frame.animate.reorient(21, 52, 0, (1.56, -1.16, 0.64), 5.60))
        Animations=get_animations(linear_comb3[7],ax)
        self.play(LaggedStart(Animations),run_time=2)
        grp,arrow=get_lines(ax,linear_comb3[7])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow))
        self.play(FadeOut(grp))

        
        # span 3d space
        self.play(linear_comb3.animate.scale(0.5,about_point=linear_comb3.get_corner(UP+RIGHT)),
            mat3.animate.scale(0.5,about_point=mat3.get_corner(LEFT+UP))) 

# customs
class MatrixCustom(Matrix):
    def __init__(self,matrix_arr,**kwargs):
        super().__init__(matrix_arr,**kwargs)
        self.nparr=matrix_arr # just easy to get matrix array
        self.to_corner(UL)
        self.col_colors()
        self.bracket_color=WHITE
        self.brackets.set_color(self.bracket_color)
    def col_colors(self):
        number_of_columns=len(self.nparr[0,:])
        self.number_of_columns=number_of_columns
        col_colors=[TEAL_B,YELLOW,BLUE,RED_A]
        self.col_colors=col_colors
        for i in range(number_of_columns):
            self.columns[i].set_color(col_colors[i])
    def get_linear_combination(self):
        coefficients=['a','b','c','d','e']
        a=Tex('a').set_color(self.col_colors[0])
        grp=VGroup(a,self.get_matrix_nth_column_vector(0))
        for i in range(self.number_of_columns-1):
            plus=Tex('+')
            tex=Tex(coefficients[i+1]).set_color(self.col_colors[i+1])
            vec=self.get_matrix_nth_column_vector(i+1)
            grp.add(plus,tex,vec)
        grp.arrange(RIGHT).to_corner(UR)
        return grp
    def get_matrix_nth_column_vector(self,nth):
        new_arr=self.nparr[:,nth:nth+1]
        new_mat=Matrix(new_arr)
        new_mat.nparr=new_arr
        new_mat.set_color(self.col_colors[nth])
        new_mat.brackets.set_color(self.bracket_color)
        return new_mat
class ThreeDAxesCustom(ThreeDAxes):
    def __init__(
        self,
        x_range = (-6.0, 6.0, 1.0),
        y_range = (-3.0, 3.0, 1.0),
        z_range = (-4.0, 4.0, 1.0),
        **kwargs
        ):
        super().__init__(x_range, y_range, z_range,**kwargs)
        # axes color
        self.x_axis.set_color(PURPLE_B)
        self.y_axis.set_color(PURPLE_B)
        self.z_axis.set_color(PURPLE_B)
        # ticks color
        self.x_axis.ticks.set_color(YELLOW)
        self.y_axis.ticks.set_color(YELLOW)
        self.z_axis.ticks.set_color(YELLOW)
        # hide the tick in origin
        self.x_axis.ticks[int(self.x_axis.x_max)].set_opacity(0)
        self.y_axis.ticks[int(self.y_axis.x_max)].set_opacity(0)
        self.z_axis.ticks[int(self.z_axis.x_max)].set_opacity(0)

    def add_coordinate_labels(self,
        x_values=None,
        y_values=None,
        z_values=None,
        excluding=[],font_size=18,**kwargs) :
        super().add_coordinate_labels(
            x_values=x_values,
            y_values=y_values,
            excluding=excluding,font_size=font_size,**kwargs) 
        z_labels = self.z_axis.add_numbers(z_values, 
            excluding=excluding,direction=LEFT,font_size=font_size,**kwargs)
        for label in z_labels:
            label.rotate(PI / 2, RIGHT)
        self.coordinate_labels.add(z_labels)
        # (3,2,-1,0,1,2,3) labels color
        self.coordinate_labels.set_color(YELLOW)
        self.set_zero_opacity()
        return self.coordinate_labels
    def set_zero_opacity(self,opacity=0):
        x_grp,y_grp,z_grp=self.coordinate_labels
        x_correct_num=-self.x_axis.x_min
        y_correct_num=-self.y_axis.x_min
        z_correct_num=-self.z_axis.x_min
        x_grp[int(0+x_correct_num)].set_opacity(opacity)
        y_grp[int(0+y_correct_num)].set_opacity(opacity)
        z_grp[int(0+z_correct_num)].set_opacity(opacity)
    def add_axis_labels(self,*args,**kwargs):
        super().add_axis_labels(*args,**kwargs,font_size=70,buff=0.3)
        # axes labels (x,y,z) color
        self.axis_labels.set_color(PURPLE_A)
class NumberPlaneCustom(NumberPlane):
    default_axis_config: dict = dict(
        stroke_color=PURPLE_A,
        stroke_width=2,
        include_ticks=False,
        include_tip=False,
        line_to_number_buff=SMALL_BUFF,
        line_to_number_direction=DL,
        )
    default_y_axis_config: dict = dict(
        line_to_number_direction=DL,
        )
    def __init__(self, x_range=(-6,6,1), y_range=(-3,3,1),
        background_line_style: dict = dict(
            stroke_color=YELLOW_A,
            stroke_width=2,
            stroke_opacity=0.5,
            ), 
        faded_line_style: dict = dict(
            stroke_color=PURPLE_A,
            stroke_width=1,
            stroke_opacity=0.4,),
         **kwargs
    ):
        super().__init__(
            x_range=x_range, y_range=y_range,
            background_line_style=background_line_style,
            faded_line_style=faded_line_style,
            **kwargs)
class nbpscene(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame

        # start
        nbp=NumberPlaneCustom()
        self.add(nbp)

        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        self.add(ax)

        frame.to_default_state()
        animations=[
                Write(ax),
                frame.animate.reorient(15, 39, 0, (0.56, 0.66, 0.32)),
                Write(nbp),
                ]
        self.play(LaggedStart(animations,lag_ratio=0.02),run_time=3)   # beautiful !!


        

        










