from manim_imports_ext import *
class starting(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        title=TextCustom(en='Data Warehouse',ch='数据仓库')
        title.scale(1.5)
        # write title
        self.play(Write(title.en),Write(title.ch))
        self.play(FadeOut(title))
        # equation
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

        # image part
        ax=ThreeDAxesCustom()
        ax.add_axis_labels()
        im=ImageMobject('resized_image.jpg')
        im_r=ImageMobject('r.png')
        im_g=ImageMobject('g.png')
        im_b=ImageMobject('b.png')
        im_grp=Group(im_r,im_g,im_b)
        positions=im_grp.copy().arrange(RIGHT)
        im_r.shift(LEFT*2)
        im_b.shift(RIGHT*2)
        im_r.rotate(-PI/4,axis=UP)
        im_g.rotate(-PI/4,axis=UP)
        im_b.rotate(-PI/4,axis=UP)
        
        for ima in im_grp:
            ima.apply_depth_test().set_opacity(0.5)
        # self.add(im,ax)
        # self.add(im_grp)
        self.play(FadeIn(im),rate_func=linear)
        self.play(im.animate.rotate(-PI/4,axis=UP))
        self.play(FadeTransform(im,im_g),FadeTransform(im.copy(),im_r),
                FadeTransform(im.copy(),im_b))
        self.play(LaggedStart(
            [im_r.animate.rotate(PI/4,axis=UP).move_to(positions[0]),
            im_g.animate.rotate(PI/4,axis=UP).move_to(positions[1]),
            im_b.animate.rotate(PI/4,axis=UP).move_to(positions[2])],lag_ratio=0.1))

        # add_image_matrix
        image_r=Image.open('r.png')
        image_g=Image.open('g.png')
        image_b=Image.open('b.png')
        r_arr=np.array(image_r.getdata())[:,0].reshape(50,50)
        g_arr=np.array(image_g.getdata())[:,1].reshape(50,50)
        b_arr=np.array(image_b.getdata())[:,2].reshape(50,50)
        mat_r=Matrix(r_arr[:15,:10],ellipses_col=9,ellipses_row=14)
        mat_g=Matrix(g_arr[:15,:10],ellipses_col=9,ellipses_row=14)
        mat_b=Matrix(b_arr[:15,:10],ellipses_col=9,ellipses_row=14)
        mat_grp=VGroup(mat_r,mat_g,mat_b)
        mat_r.move_to(im_r).match_width(im_r).set_color(RED)
        mat_g.move_to(im_g).match_width(im_r).set_color(GREEN)
        mat_b.move_to(im_b).match_width(im_r).set_color("#6666FF")
        # self.add(mat_grp)
        self.play(LaggedStart(
            [FadeTransform(im_r,mat_r),
            FadeTransform(im_g,mat_g),
            FadeTransform(im_b,mat_b)],lag_ratio=0.5))
        self.play(FadeOut(mat_grp),run_time=1)

        # Builder of Spaces
        title2=TextCustom(en='Builder of Spaces',ch='空间建构师')
        title2.scale(1.5)
        # self.add(title2)
        self.play(Write(title2.en),Write(title2.ch))
        self.play(FadeOut(title2))

        # 2d space
        mat2d=MatrixCustom(np.array([[1,0],[0,1]]))
        ax=ThreeDAxesCustom()
        ax.add_coordinate_labels()
        ax.add_axis_labels()
        lb2d=mat2d.get_linear_combination()
        arrow1,arrow2=mat2d.get_column_arrows(ax)
        change_parts=mat2d.get_changeable_parts()

        # 2d animation
        mat2d.save_state()
        mat2d.center().scale(2)
        self.play(Write(mat2d))
        self.play(mat2d.animate.restore())
        animations=[AnimationGroup(TransformFromCopy(mat2d.brackets,
                mat2d.vector_matrices[0].brackets,path_arc=1),
            TransformFromCopy(mat2d.get_column(0),
                mat2d.vector_matrices[0].get_column(0),path_arc=1)),
                    AnimationGroup(TransformFromCopy(mat2d.brackets,
                mat2d.vector_matrices[1].brackets,path_arc=1),
            TransformFromCopy(mat2d.get_column(1),
                mat2d.vector_matrices[1].get_column(0),path_arc=1))]
        self.play(LaggedStart(*animations,lag_ratio=0.5))
        self.play(Write(mat2d.parts),run_time=2)
        ax.x_axis.set_opacity(0.5)
        ax.y_axis.set_opacity(0.5)
        self.play(Write(ax.x_axis),Write(ax.y_axis))
        # self.play(TransformFromCopy(mat2d.vector_matrices[0].get_column(0),arrow1))
        self.play(LaggedStart(
         [TransformFromCopy2(mat2d.vector_matrices[0].elements[0],ax.x_axis.numbers[6].copy()),
         ShrinkToPoint(mat2d.vector_matrices[0].elements[1].copy(),point=ax.c2p(0,0,0))],
         lag_ratio=0.5))
        self.play(GrowArrow(arrow1))
        self.play(LaggedStart(
         [ShrinkToPoint(mat2d.vector_matrices[1].elements[0].copy(),point=ax.c2p(0,0,0)),
         TransformFromCopy2(mat2d.vector_matrices[1].elements[1],ax.y_axis.numbers[3].copy()),]
         ,lag_ratio=0.5))
        self.play(GrowArrow(arrow2))
        # span
        nbp=NumberPlaneCustom()
        arrow3=get_added_arrow([arrow1,arrow2],ax)
        changeable_parts=mat2d.get_changeable_parts()
        self.play(*map(FlashAround,mat2d.parts))
        self.play(ReplacementTransform(mat2d.parts,changeable_parts))
        self.play(TransformFromCopy(arrow1,arrow3),TransformFromCopy(arrow2,arrow3),
            arrow1.animate.set_opacity(0.5),arrow2.animate.set_opacity(0.5))
        vt1=ValueTracker(1)
        vt2=ValueTracker(1)
        changeable_parts[0].always.set_color(mat2d.color_palette[0])
        changeable_parts[1].always.set_color(mat2d.color_palette[1])
        changeable_parts[0].f_always.set_value(lambda:vt1.get_value())
        changeable_parts[1].f_always.set_value(lambda:vt2.get_value())
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
        def get_span2d_animation(x,y,nbp):
            return [
            AnimationGroup(vt1.animate.set_value(x),vt2.animate.set_value(y)),
            AnimationGroup(Write(nbp.index_lines(x,y)),nbp.index_lines(x,y).animate.set_opacity(0.3))]
        def get_span2d_animation2(x,y,nbp):
            return [
            AnimationGroup(vt1.animate.set_value(x),vt2.animate.set_value(y)),
            AnimationGroup(Write(nbp.index_lines(x,y)[1]),
                nbp.index_lines(x,y)[1].animate.set_opacity(0.3))]
        arrow1.add_updater(get_updater(arrow1.nparr,vt1,ax))
        arrow2.add_updater(get_updater(arrow2.nparr,vt2,ax))
        arrow3.add_updater(get_added_arrow_updater(vt1,vt2,arrow1.nparr,arrow2.nparr,ax))
        self.play(LaggedStart(get_span2d_animation(-2,-2,nbp)))
        self.play(LaggedStart(get_span2d_animation(-5,1,nbp)))
        self.play(LaggedStart(get_span2d_animation(1,3,nbp)))
        self.play(LaggedStart(get_span2d_animation(4,-3,nbp)),run_time=0.9)
        self.play(LaggedStart(get_span2d_animation(-6,2,nbp)),run_time=0.9)
        self.play(LaggedStart(get_span2d_animation(2,-1,nbp)),run_time=0.8)
        self.play(LaggedStart(get_span2d_animation(-4,0,nbp)),run_time=0.8)

        self.play(LaggedStart(get_span2d_animation2(3,3,nbp)),run_time=0.7)
        self.play(LaggedStart(get_span2d_animation2(-3,2,nbp)),run_time=0.7)
        self.play(LaggedStart(get_span2d_animation2(5,1,nbp)),run_time=0.6)
        self.play(LaggedStart(get_span2d_animation2(-1,-1,nbp)),run_time=0.6)
        self.play(LaggedStart(get_span2d_animation2(6,-2,nbp)),run_time=0.6)
        self.play(LaggedStart(get_span2d_animation2(0,0,nbp)),run_time=0.6)

        self.play(Write(nbp.faded_lines.set_opacity(0.4)),
            nbp.background_lines.animate.set_opacity(0.5))
        self.play(ax.x_axis.animate.set_opacity(1),ax.y_axis.animate.set_opacity(1),)

def get_added_arrow(arrows,axis):
    coord=np.zeros(3)
    for arrow in arrows:
        coord += np.array(axis.p2c(arrow.get_end())) 
    added_arrow=Arrow(axis.c2p(0,0,0),axis.c2p(*coord),buff=0)
    colors = [arrow.get_color() for arrow in arrows]
    added_arrow.set_color(average_color(*colors))
    return added_arrow

class ShrinkToPoint(Transform):
    def __init__(self,mobject,point,**kwargs):
        mobject=mobject
        target_mobject=mobject.copy().scale(0,about_point=point)
        super().__init__(mobject,target_mobject,remover=True,**kwargs)

class TransformFromCopy2(Transform):
    def __init__(self, mobject, target_mobject, **kwargs):
        super().__init__(mobject.copy(), target_mobject.copy(), **kwargs)
    def clean_up_from_scene(self, scene: Scene) -> None:
        scene.remove(self.mobject)
        scene.remove(self.target_mobject)



class data_warehouse(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        
        pass
class builder_of_space(InteractiveScene):
     def construct(self):
         # init
         frame=self.frame
         # start
class magician_of_transformation(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        
        pass
class ending(InteractiveScene):
    def construct(self):
        # init
        frame=self.frame
        # start
        
        pass