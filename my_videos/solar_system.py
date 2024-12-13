from manimlib import *
import re
import numpy as np
from PIL import Image
from manimlib.utils.images import get_full_raster_image_path
def get_orthogonal_proj_matrix(A):
    proj_mat=np.dot(np.dot(A,np.linalg.inv(np.dot(A.T,A))),A.T)
    return proj_mat
# opengl image blending???? how to do this??
class image(InteractiveScene):
    def construct(self):
        # init
        # state=self.checkpoint_states
        frame=self.frame
        # start
        image=Image.open('kun.png')
        resized_image=image.resize((50,50))
        resized_image.save('show_image.jpg')
        def save_rgb_image(image):
            data=image.getdata()
            r = [(d[0], 0, 0) for d in data]
            g = [(0, d[1], 0) for d in data]
            b = [(0, 0, d[2]) for d in data]
            image.putdata(r)
            image.save('r.png')
            image.putdata(g)
            image.save('g.png')
            image.putdata(b)
            image.save('b.png')
        save_rgb_image(resized_image)
        image_r=Image.open('r.png')
        image_g=Image.open('g.png')
        image_b=Image.open('b.png')
        r_arr=np.array(image_r.getdata())[:,0].reshape(50,50)
        g_arr=np.array(image_g.getdata())[:,1].reshape(50,50)
        b_arr=np.array(image_b.getdata())[:,2].reshape(50,50)

        # add image
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        self.add(ax)
        frame.reorient(58, 64, 0, (0.22, 1.1, 0.42))
        im_mob=ImageMobject('show_image.jpg')
        im_mob.rotate(PI/2,axis=RIGHT)
        im_mob.set_opacity(0.5)
        im_mob_r=ImageMobject('r.png')
        im_mob_g=ImageMobject('g.png')
        im_mob_b=ImageMobject('b.png')
        grp_rgb=Group(im_mob_r,im_mob_g,im_mob_b).arrange(OUT,buff=3)
        grp_rgb.rotate(PI/2,axis=RIGHT)
        grp_rgb.center()
        im_mob_r.apply_depth_test()
        im_mob_g.apply_depth_test()
        im_mob_b.apply_depth_test()
        grp_rgb.set_opacity(0.5)
        im_mob.scale(2)
        grp_rgb.scale(2)
        # image_grp=Group(grp_rgb,im_mob).arrange(RIGHT)
        self.add(grp_rgb)
        frame.reorient(25, 64, 0, (0.38, 1.45, 0.72), 13.31)

        # Matrix
        mat_r=Matrix(r_arr[:10,:10],ellipses_col=9,ellipses_row=9)
        mat_g=Matrix(g_arr[:10,:10],ellipses_col=9,ellipses_row=9)
        mat_b=Matrix(b_arr[:10,:10],ellipses_col=9,ellipses_row=9)
        grp_mat=VGroup(mat_r,mat_g,mat_b).arrange(OUT,buff=1)
        grp_mat.rotate(PI/2,axis=RIGHT)
        self.add(grp_mat)
        mat_r.match_y(im_mob_r)
        mat_g.match_y(im_mob_g)
        mat_b.match_y(im_mob_b)
        # mat.match_height()

        # animation
        frame.to_default_state()
        im_mob.center()
        grp_rgb.center()
        mat_r.match_width(im_mob_r)
        im_mob_r.get_corner(UL)
        mat_r.get_corner(UL)






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
        text1=TextCustom('Matrix','矩阵',DOWN,buff=0.4).next_to(tex_mat_aug,UP)
        text2=TextCustom('Augmented Matrix','增广矩阵',DOWN,buff=0.2).next_to(tex_mat_aug,UP)
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
        a=Tex(R'a')
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
        grp,arrow1=get_lines(ax,linear_comb3[1])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow1))
        self.play(FadeOut(grp))

        self.play(frame.animate.reorient(-35, 59, 0, (-0.22, 0.08, 0.91), 4.69))
        Animations=get_animations(linear_comb3[4],ax)
        self.play(LaggedStart(Animations),run_time=2)
        grp,arrow2=get_lines(ax,linear_comb3[4])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow2))
        self.play(FadeOut(grp))

        self.play(frame.animate.reorient(21, 52, 0, (1.56, -1.16, 0.64), 5.60))
        Animations=get_animations(linear_comb3[7],ax)
        self.play(LaggedStart(Animations),run_time=2)
        grp,arrow3=get_lines(ax,linear_comb3[7])
        self.play(LaggedStartMap(ShowCreation,grp[0],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[1],run_time=1))
        self.play(LaggedStartMap(ShowCreation,grp[2],run_time=1),GrowArrow(arrow3))
        self.play(FadeOut(grp))

        
        # span 3d space
        def get_rotmat(x2:np.ndarray,y2:np.ndarray,x1=RIGHT,y1=UP,z1=OUT):
            # x2,y2 just need to be two vectors in that plane
            # x2,y2 can be not orthogonal
            # x2=np.array([1,2,3]) form
            # make orthonomal matrix
            x2=x2
            y2=y2-x2*(np.dot(y2,x2.T)/np.dot(x2,x2.T))
            z2=cross(x2,y2)
            init_mat=np.array([normalize(x1),normalize(y1),normalize(z1)]) # A'
            final_mat=np.array([normalize(x2),normalize(y2),normalize(z2)]) # B'
            return np.dot(init_mat,final_mat.T) # R=A'B

        self.play(linear_comb3.animate.scale(0.5,about_point=linear_comb3.get_corner(UP+RIGHT)),
            mat3.animate.scale(0.5,about_point=mat3.get_corner(LEFT+UP))) 
        nbp=NumberPlaneCustom()
        self.play(Write(nbp),frame.animate.reorient(-33, 52, 0, (0.97, -0.42, 1.79), 5.60))

        rotmat=get_rotmat(arrow1.get_end(),arrow2.get_end())
        self.play(nbp.animate.apply_matrix(rotmat))

        # spane
        def get_span_along_certain_direction(plane,direction,number=3,buff=1):
            direction=normalize(direction)
            copys1=plane.replicate(math.ceil(number/2))
            for index,submob in enumerate(copys1.submobjects):
                index=index+1
                submob.shift(direction*buff*index)
            copys2=plane.replicate(math.floor(number/2))
            for index,submob in enumerate(copys2.submobjects):
                index=index+1
                submob.shift(-direction*buff*index)
            grp=VGroup(copys1,copys2)
            return grp
        # faded_nbp=NumberPlaneCustom(
        #     background_line_style=dict(stroke_color=PURPLE_A,
        #     stroke_width=1,
        #     stroke_opacity=0.4),
        #     axis_config=dict(stroke_color=PURPLE,
        #     stroke_width=2,
        #     stroke_opacity=0.8))      
        # nbp.set_style(stroke_width=1,stroke_color=WHITE,stroke_opacity=0.5,
        #     fill_color=WHITE,fill_opacity=0.5)
        grp=get_span_along_certain_direction(nbp,arrow3.get_end(),number=10)
        self.add(grp)
        
        # color
        text=Text('check color gradient').set_color_by_gradient([BLUE,GREEN])
        self.add(text)
        ax=NumberLine()
        self.add(ax)
        ax.space_out_submobjects()
class span_2d_space(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        mat=MatrixCustom(np.array([[1,0],[0,1],[1,1]]))
        grp=mat.get_linear_combination()
        self.add(mat,grp)
        changeable_parts=mat.get_changeable_parts()
        mat.fix_in_frame()
        grp.fix_in_frame()
        changeable_parts.fix_in_frame()
        # spane space
        

        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        self.add(ax)
        frame.reorient(17, 35, 0, (0.49, 0.6, 0.23), 8.00)
        
        # arrows
        arrow1,arrow2=mat.get_column_vectors(ax)
        self.add(arrow1,arrow2)
        arrow3=always_redraw(lambda:get_added_arrow(arrow1,arrow2,ax))
        lines=always_redraw(lambda:get_dashed_lines(arrow1,arrow2,arrow3))
        self.add(arrow3,lines)

        # decimal number
        self.remove(mat.parts)
        self.add(changeable_parts)
        arrow1_arr=mat.vector_matrices[0].nparr.flatten()
        arrow2_arr=mat.vector_matrices[1].nparr.flatten()
        factor1=changeable_parts[0]
        factor2=changeable_parts[1]
        vt1=ValueTracker(1)
        vt2=ValueTracker(1)
        def arrow1_updater(mob):
            arrow1_coord=vt1.get_value()*arrow1_arr
            mob.put_start_and_end_on(ax.c2p(0,0,0),ax.c2p(*arrow1_coord))        
        def arrow2_updater(mob):
            arrow2_coord=vt2.get_value()*arrow2_arr
            mob.put_start_and_end_on(ax.c2p(0,0,0),ax.c2p(*arrow2_coord))
        arrow1.add_updater(arrow1_updater)
        arrow2.add_updater(arrow2_updater)
        factor1.f_always.set_value(lambda:vt1.get_value())
        factor1.always.set_color(TEAL)
        factor2.f_always.set_value(lambda:vt2.get_value())
        factor2.always.set_color(YELLOW)
        self.play(vt1.animate.set_value(3),vt2.animate.set_value(2))
        self.play(vt1.animate.set_value(-3),vt2.animate.set_value(3))
        self.play(vt1.animate.set_value(-3),vt2.animate.set_value(-2))
        self.play(vt1.animate.set_value(3),vt2.animate.set_value(-3))

        arrow3.clear_updaters()
        lines.clear_updaters()
        factor1.clear_updaters()
        factor2.clear_updaters()
        arrow1.clear_updaters()
        arrow2.clear_updaters()

class span_2d_space_animation(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        ax.add_coordinate_labels()
        nbp=NumberPlaneCustom()
        ax.set_opacity(0.5)
        self.add(ax.x_axis,ax.y_axis)

        # matrix,arrows 
        mat=MatrixCustom(np.array([[1,0],[0,1]]))
        comb=mat.get_linear_combination()
        mat.fix_in_frame()
        comb.fix_in_frame()
        arrow1,arrow2=mat.get_column_arrows(ax)
        added_arrow=get_added_arrow([arrow1,arrow2],axis=ax)
        self.add(mat,comb,arrow1,arrow2)
        self.add(added_arrow) 

        # chaneable parts
        frame.to_default_state()
        changeable_parts=mat.get_changeable_parts()
        changeable_parts.fix_in_frame()
        self.remove(mat.parts)
        self.add(changeable_parts)

        # updater
        def get_updater(arr,vt,ax):
            def updater(m):
                factor=vt.get_value()
                m.put_start_and_end_on(ax.c2p(0,0,0),ax.c2p(*factor*arr)) 
            return updater
        def get_added_arrow_updater(vt1,vt2,arr1,arr2,ax):
            def updater(m):
                factor1=vt1.get_value()
                factor2=vt2.get_value()
                arr=vt1.get_value()*arr1+vt2.get_value()*arr2
                m.put_start_and_end_on(ax.c2p(0,0,0),ax.c2p(*arr))
            return updater
        vt1=ValueTracker(1)
        vt2=ValueTracker(1)
        changeable_parts[0].always.set_color(TEAL_B)
        changeable_parts[1].always.set_color(YELLOW)
        changeable_parts[0].f_always.set_value(lambda:vt1.get_value())
        changeable_parts[1].f_always.set_value(lambda:vt2.get_value())
        arrow1.add_updater(get_updater(arrow1.nparr,vt1,ax))
        arrow2.add_updater(get_updater(arrow2.nparr,vt2,ax))
        added_arrow.add_updater(get_added_arrow_updater(vt1,vt2,arrow1.nparr,arrow2.nparr,ax))
        dashedline1=always_redraw(lambda:
            DashedLine(arrow1.get_end(),added_arrow.get_end(),stroke_opacity=0.8))
        dashedline2=always_redraw(lambda:
            DashedLine(arrow2.get_end(),added_arrow.get_end(),stroke_opacity=0.8))
        self.add(dashedline1,dashedline2)

        # loop animations
        tt=TracingTail(lambda:added_arrow.get_end(),time_traced=0.2)
        self.add(tt)
        anims=get_span_animation(vt1=vt1,vt2=vt2,axis=ax)
        for i,ani in enumerate(anims,start=1):
            self.play(eval(ani),rate_func=linear,
                run_time=1/math.sqrt(i))
        arrow1.suspend_updating()
        arrow2.suspend_updating()
        added_arrow.suspend_updating()
        dashedline1.suspend_updating()
        dashedline2.suspend_updating()
        self.play(Write(nbp),
            LaggedStartMap(FadeOut,VGroup(tt,arrow1,arrow2,added_arrow,dashedline1,dashedline2)))
        self.play(ax.x_axis.animate.set_opacity(1),ax.y_axis.animate.set_opacity(1))

        # span 3d space
        self.remove(mat,changeable_parts,comb)
        mat_3d=MatrixCustom(np.array([[1,0,0],[0,1,0],[0,0,1]]))
        mat_3d.fix_in_frame()
        mat_3d_comb=mat_3d.get_linear_combination()
        mat_3d_comb.fix_in_frame()
        self.add(mat_3d,mat_3d_comb)
        mat_3d.scale(0.7,about_point=mat_3d.get_corner(UL))
        mat_3d_comb.scale(0.7,about_point=mat_3d_comb.get_corner(UR))
        changeable_parts_3d=mat_3d.get_changeable_parts(font_size=20)
        changeable_parts_3d.fix_in_frame()
        self.play(*map(FlashAround,mat_3d.parts))
        self.play(ReplacementTransform(mat_3d.parts,changeable_parts_3d))
        self.add(ax.z_axis)
        arrows=mat_3d.get_column_arrows(ax)
        added_arrow2=get_added_arrow(arrows,axis=ax)
        self.add(arrows)
        self.add(added_arrow2)
        # self.remove(mat_3d.parts)
        # self.add(changeable_parts_3d)
class numberspcae(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        self.add(ax)

        # lines
        def get_lines_grp(ax):
            x_min=ax.x_axis.x_min
            x_max=ax.x_axis.x_max
            y_min=ax.y_axis.x_min
            y_max=ax.y_axis.x_max
            z_min=ax.z_axis.x_min
            z_max=ax.z_axis.x_max
            step=1
            # x-lines
            x_grp=VGroup()
            for z in np.arange(z_min,z_max+1,step):
                x_subgrp=VGroup()
                for y in np.arange(y_min,y_max+1,step):
                    line=Line(ax.c2p(x_min,y,z),ax.c2p(x_max,y,z))
                    x_subgrp.add(line)
                x_subgrp.set_opacity(0.2)
                x_grp.add(x_subgrp)
            # y-lines
            y_grp=VGroup()
            for z in np.arange(z_min,z_max+1,step):
                y_subgrp=VGroup()
                for x in np.arange(x_min,x_max+1,step):
                    line=Line(ax.c2p(x,y_min,z),ax.c2p(x,y_max,z))
                    y_subgrp.add(line)
                y_subgrp.set_opacity(0.2)
                y_grp.add(y_subgrp)
            # z-lines
            z_grp=VGroup()
            for y in np.arange(y_min,y_max+1,step):
                z_subgrp=VGroup()
                for x in np.arange(x_min,x_max+1,step):
                    line=Line(ax.c2p(x,y,z_min),ax.c2p(x,y,z_max))
                    z_subgrp.add(line)
                z_subgrp.set_opacity(0.2)
                z_grp.add(z_subgrp)
            xyz_grp=VGroup(x_grp,y_grp,z_grp)
            return xyz_grp

        grp=get_lines_grp(ax)
        # (x,y,z)=(1,1,1)
        # x-x_min,y-y_min,z-z_min
        def grp_index(ax,grp,target_xyz):
            x=target_xyz[0]
            y=target_xyz[1]
            z=target_xyz[2]
            x_min=ax.x_axis.x_min
            y_min=ax.y_axis.x_min            
            z_min=ax.z_axis.x_min
            z_index=int(z-z_min)
            x_index=int(x-x_min)
            y_index=int(y-y_min)
            x_line=grp[0][z_index][y_index]
            y_line=grp[1][z_index][x_index]
            z_line=grp[2][y_index][x_index]
            return x_line,y_line,z_line


        # self.add(grp)
        self.add(*grp_index(ax,grp,[-6,1,1]))
        for x in np.arange(-6,6,1):
            self.play(ReplacementTransform(
                VGroup(*grp_index(ax,grp,[x,1,1])),
                VGroup(*grp_index(ax,grp,[x+1,1,1])),))

        # text
        text=TextCustom(en='Matrix',ch='矩阵')
        text2=TextCustom(en='Matrix',ch='矩阵',font_en='WenCang',font_ch='WenCang')
        VGroup(text,text2).arrange(RIGHT)
        self.add(text,text2)




        

def get_span_animation(vt1,vt2,axis):
    x_min=axis.x_axis.x_min
    x_max=axis.x_axis.x_max    
    y_min=axis.y_axis.x_min
    y_max=axis.y_axis.x_max
    max_number=max(x_max,y_max,abs(x_min),abs(y_min))
    animations=[]
    for i in range(int(max_number)):
        value=i+1
        animations.append(f'vt1.animate.set_value({min(value,x_max)})')
        animations.append(f'vt2.animate.set_value({min(value,y_max)})')
        animations.append(f'vt1.animate.set_value({max(-value,x_min)})')
        animations.append(f'vt2.animate.set_value({max(-value,y_min)})')     
    return animations

def get_added_arrow(arrows,axis):
    coord=np.zeros(3)
    for arrow in arrows:
        coord += np.array(axis.p2c(arrow.get_end())) 
    added_arrow=Arrow(axis.c2p(0,0,0),axis.c2p(*coord),buff=0)
    return added_arrow

# customs
class TextCustom(VGroup):
    def __init__(self, 
        en=None                   ,ch=None,
        direction=DOWN,
        buff=0.2,
        font_en="Mongolian Baiti" ,font_ch="SJsuqian",
        font_size_en=48           ,font_size_ch=40,
        en_config=dict()          ,ch_config=dict(),
        **kwargs):
        super().__init__(**kwargs)
        self.en = None
        self.ch = None
        if en is not None:
            self.en = Text(en, font=font_en, font_size=font_size_en,**en_config)
            self.add(self.en)
        if ch is not None:
            self.ch = Text(ch, font=font_ch, font_size=font_size_ch,**ch_config)
            self.add(self.ch)
        if self.en and self.ch:
            self.ch.next_to(self.en, direction, buff=buff)

class MatrixCustom(Matrix):
    def __init__(self,matrix_arr,**kwargs):
        super().__init__(matrix_arr,**kwargs)
        # attributes
        self.nparr=matrix_arr # just easy to get matrix array
        self.number_of_columns=len(self.nparr[0,:])
        self.color_palette=[TEAL_B,YELLOW,BLUE,RED_A]
        # position
        self.to_corner(UL)
        # colors
        self.set_col_colors()
        self.bracket_color=WHITE
        self.brackets.set_color(self.bracket_color)
    def set_col_colors(self):
        for i in range(self.number_of_columns):
            self.columns[i].set_color(self.color_palette[i])
    def get_linear_combination(self,**kwargs):
        # 
        coefficients=['a','b','c','d','e']
        a=Tex('a').set_color(self.color_palette[0])
        first_vector=self.get_matrix_nth_column_vector(0)
        vector_matrices=self.get_all_column_vectors()
        grp=VGroup(a,vector_matrices[0])
        self.parts=VGroup(a)
        for i in range(self.number_of_columns-1):
            plus=Tex('+')
            tex=Tex(coefficients[i+1]).set_color(self.color_palette[i+1])
            vec=vector_matrices[i+1]
            grp.add(plus,tex,vec)
            self.parts.add(VGroup(plus,tex))
        grp.arrange(RIGHT,**kwargs).to_corner(UR)
        self.vector_matrices=vector_matrices
        self.linear_combination=grp
        return self.linear_combination
    def get_changeable_parts(self,places=1,font_size=30,first_buff=0.2,inner_buff=0.1):
        number_of_parts=len(self.parts)
        changeable_parts=VGroup()
        for i in range(number_of_parts):
            if i == 0 :
                number=DecimalNumber(1,num_decimal_places=places,include_sign=True,font_size=font_size)
                VGroup(number[0],number[1:]).arrange(RIGHT,buff=inner_buff)
                number[0].set_color(WHITE)
                number[1:].match_color(self.parts[i])
                number.move_to(self.parts[i]).shift(LEFT*first_buff)
            else :
                number=DecimalNumber(1,num_decimal_places=places,include_sign=True,font_size=font_size)
                VGroup(number[0],number[1:]).arrange(RIGHT,buff=inner_buff)
                number[0].match_color(self.parts[i][0])
                number[1:].match_color(self.parts[i][1])
                number.move_to(self.parts[i])
            changeable_parts.add(number)
        self.changeable_parts=changeable_parts
        return self.changeable_parts
    def get_all_column_vectors(self):
        grp=VGroup()
        for i in range(self.number_of_columns):
            grp.add(self.get_matrix_nth_column_vector(i))
        grp.arrange(RIGHT)
        return grp
    def get_matrix_nth_column_vector(self,nth):
        new_arr=self.nparr[:,nth:nth+1]
        new_mat=MatrixCustom(new_arr)
        new_mat.nparr=new_arr
        new_mat.set_color(self.color_palette[nth])
        new_mat.brackets.set_color(self.bracket_color)
        return new_mat
    def get_column_arrows(self,ax,**kwargs):
        grp=VGroup()
        for i in range(self.number_of_columns):
            arrow=Arrow(ax.c2p(0,0,0),ax.c2p(*self.nparr[:,i]),buff=0,**kwargs)
            arrow.nparr=self.nparr[:,i]
            arrow.match_color(self.columns[i])
            grp.add(arrow)
        return grp



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
        # remove the tick in origin
        self.remove(self.x_axis.ticks[int(self.x_axis.x_max)])
        self.remove(self.y_axis.ticks[int(self.y_axis.x_max)])
        self.remove(self.z_axis.ticks[int(self.z_axis.x_max)])

    def add_coordinate_labels(self,
        x_values=None,
        y_values=None,
        z_values=None,
        excluding=[0],font_size=18,**kwargs) :
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
        # self.set_zero_opacity()
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
        self.axis_labels.set_color(YELLOW)
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
    def __init__(self, 
        x_range=(-6,6,1), 
        y_range=(-3,3,1),
        background_line_style: dict = dict(
            stroke_color=YELLOW_A,
            stroke_width=2,
            stroke_opacity=0.3,
            ), 
        faded_line_style: dict = dict(
            stroke_color=PURPLE_B,
            stroke_width=1,
            stroke_opacity=0.3,),
         **kwargs
    ):
        super().__init__(
            x_range=x_range, y_range=y_range,
            background_line_style=background_line_style,
            faded_line_style=faded_line_style,
            **kwargs)


        

        










