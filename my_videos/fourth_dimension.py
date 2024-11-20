from manimlib import *
import numpy as np
from scipy.spatial.transform import Rotation

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
def get_projection_point(point,frame_center,camera_postion): # perspective 
    #convert list to numpy comlumn vector 
    frame_center=np.array([frame_center]).T
    point=np.array([point]).T
    camera_postion=np.array([camera_postion]).T
    # codes:
    point=point-frame_center
    n=camera_postion-frame_center 
    proj_mat=np.dot(n,n.T)/np.dot(n.T,n)
    proj_point=np.dot(proj_mat,point)
    dirc_vector=point-proj_point
    scale_factor=get_norm(n)/(get_norm(n-proj_point))
    target_point=scale_factor*dirc_vector
    final_point=target_point+frame_center
    return final_point.T[0]   # column --> row --> first row (a list)

    
class matrix(InteractiveScene):
    def construct(self):
        # start
        # Scene()
        # CameraFrame()
        # Mobject()
        # test function
        frame=self.frame
        ax=ThreeDAxes((-6,6),(-3,3))
        ax.add_coordinate_labels()
        ax.add_axis_labels(font_size=60)
        rec=Rectangle(FRAME_WIDTH,FRAME_HEIGHT,fill_opacity=0.5)
        self.add(rec,ax)
        self.play(frame.animate.reorient(50, 30, 0))
        camera_position=frame.get_implied_camera_location()
        mat=get_rotation_matrix(camera_position)
        angles=Rotation.from_matrix(mat).as_euler('xyz')
        self.play(rec.animate.apply_matrix(mat))
        ax2=ax.copy()
        self.play(frame.animate.set_orientation(Rotation.from_matrix(mat)),run_time=2)

        # way1
        # self.play(ax2.animate.rotate(angles[0],axis=RIGHT,about_point=ORIGIN))
        # self.play(ax2.animate.rotate(angles[1],axis=UP,about_point=ORIGIN))
        # self.play(ax2.animate.rotate(angles[2],axis=OUT,about_point=ORIGIN))

        # way2
        self.play(ax2.animate.apply_matrix(mat).make_smooth()) 

        # delet y,z,rec
        self.remove(ax2[1],ax2[2],rec)
        self.play(frame.animate.reorient(50, 30, 0))

        # perspective proj
        fc=frame.get_center()
        cp=frame.get_implied_camera_location()
        def projection_wrapper(point):
            point=point
            fc=frame.get_center()
            cp=frame.get_implied_camera_location()
            return get_projection_point(point,fc,cp)
        self.play(ax.animate.apply_function(
            projection_wrapper))



class tex_in_3D(InteractiveScene):
    def construct(self):
        # start
        ax=ThreeDAxes()
        tex=Tex("x")
        tex2=Tex("x").set_color(RED).shift(RIGHT)
        frame=self.frame
        frame.reorient(0, 0, 0, (0,0,0), 2)
        self.add(ax,tex,tex2)
        

        # apply matrix
        frame.reorient(50, 30, 0)
        camera_position=frame.get_implied_camera_location()
        mat=get_rotation_matrix(camera_position)
        print("mat is {}".format(mat))
        self.play(frame.animate.reorient(-42, 60, 0, (0,0,0), 2.83),run_time=2)
        self.play(tex.animate.apply_matrix(mat)) #也许和底层的渲染方式有关把？？

        # use euler_angles
        angles=Rotation.from_matrix(mat).as_euler('xyz')
        self.play(tex2.animate.rotate(angles[0],axis=RIGHT))
        self.play(tex2.animate.rotate(angles[1],axis=UP))
        self.play(tex2.animate.rotate(angles[2],axis=OUT))

        # make smooth
        self.play(tex.animate.make_smooth())

        