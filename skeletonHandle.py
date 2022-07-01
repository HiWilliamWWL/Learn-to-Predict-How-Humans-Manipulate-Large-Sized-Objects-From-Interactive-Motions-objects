import numpy as np
from scipy.spatial.transform import Rotation as Rot
from numpy import linalg as LA

class singleJoint:
    def __init__(self, num, init_pose, joints_Tpose_norm):
        self.num = num
        self.pose = init_pose   #global pose, not maintained after construction

        self.children = []   #singleJoint class object
        self.child_Tpose_norm = {}
        self.local_frame = Rot.from_euler("xyz", [.0, .0, .0]) #local frame
        self.pose_to_parent = np.array([.0, .0, .0])  #local pose
        self.parent = None
        self.joints_Tpose_norm = joints_Tpose_norm #local_T_pose_map
        
    
    def add_child(self, child_singleJoint, num_bone):
        self.children.append(child_singleJoint)
        child_num = child_singleJoint.num

        self.child_Tpose_norm[child_num] = np.array(self.joints_Tpose_norm[num_bone])  #keep it here

        child_singleJoint.pose_to_parent = child_singleJoint.pose - self.pose
        child_singleJoint.parent = self
    
    def set_local_frame(self, rot):
        self.local_frame = rot # * self.local_frame
        #print(self.child_Tpose_norm)

    
    def apply_rot(self, rot):
        assert self.local_frame is not None, "the local frame of this joint is None"
        self.local_frame = rot #* self.local_frame
        for child in self.children:
            child.pose = self.pose + self.local_frame.apply(child.pose_to_parent)
            child.apply_rot(child.local_frame)
    
    @staticmethod
    def bone_vec_norm(child):
        bone_vec = child.pose_to_parent
        bone_vec_norm = bone_vec / LA.norm(bone_vec)
        return bone_vec_norm

class JointsInfo:
    joints_connection = [(0, 1), (1, 2), (2, 3), (3, 4), 
                                    (2, 5), (5, 6), (6, 7), (7, 8), 
                                    (2, 9), (9, 10), (10, 11), (11, 12),
                                    (0, 13), (13, 14), (14, 15), (15, 19), 
                                    (0, 16), (16, 17), (17, 18), (18, 20)]

    joints_Tpose_norm = [[.0, 1.0, .0], [.0, 1.0, .0], [.0, 1.0, .0], [.0, 1.0, .0], 
                                    [0.1736482, 0.984808, .0], [1.0, .0, .0], [1.0, .0, .0], [1.0, .0, .0], 
                                    [-0.1736482, 0.984808, .0], [-1.0, .0, .0], [-1.0, .0, .0], [-1.0, .0, .0],
                                    [1.0, .0, .0], [.0, -1.0, .0], [.0, -1.0, .0], [.0, .0, 1.0], 
                                    [-1.0, .0, .0], [.0, -1.0, .0], [.0, -1.0, .0], [.0, .0, 1.0]]
    def __init__(self, init_joints_poses, global_trans = None, init_rot = None):
        #init_joints_poses.shape = (21,3) at T_0
        
        self.init_joints_poses = np.array(init_joints_poses)
        self.all_joints = []
        hip_vec = init_joints_poses[13] - init_joints_poses[16]   #orignal 13 - 16
        hip_vec = hip_vec / np.linalg.norm(hip_vec)
        back_vec = init_joints_poses[1] - init_joints_poses[0]
        back_vec = back_vec / np.linalg.norm(back_vec)
        #print(np.array([hip_vec, back_vec, np.cross(hip_vec, back_vec)]))
        #hip_vec = [.0, .0, -1]
        #back_vec = [.0, 1.0, .0]
        #print(np.array([hip_vec, back_vec, np.cross(hip_vec, back_vec)]))
        #print()
        self.init_global_rot = Rot.align_vectors(np.array([hip_vec, back_vec, np.cross(hip_vec, back_vec)]), np.array([[1.0, .0, .0], [.0, 1.0, .0], [.0, .0, 1.0]]))[0]
        #print(self.init_global_rot.as_euler("xyz", degrees=True))
        #exit()
        if init_rot is not None:
            self.init_global_rot = Rot.from_euler("y", init_rot, degrees=True)
        for i in range(21):
            joint = singleJoint(i, self.init_joints_poses[i], self.joints_Tpose_norm)
            self.all_joints.append(joint)
        for num_bone, bone in enumerate(self.joints_connection):
            start, end = bone
            self.all_joints[start].add_child(self.all_joints[end], num_bone)
        '''
        angle_2_5 = self.all_joints[5].pose - self.all_joints[2].pose
        angle_2_5 = angle_2_5 / np.linalg.norm(angle_2_5)
        angle_2_9 = self.all_joints[9].pose - self.all_joints[2].pose
        angle_2_9 = angle_2_9 / np.linalg.norm(angle_2_9)
        shoulder_angle = self.rotation_matrix_from_vectors(angle_2_5, angle_2_9)
        shoulder_angle = Rot.from_matrix(shoulder_angle)
        print(shoulder_angle.as_euler("xyz", degrees=True))
        exit()
        '''
        self.construct_rot()
        self.forward_kinematics_21Joints(self.get_all_rots())
        if global_trans is not None:
            self.apply_global_trans(global_trans)
    
    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        #a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        a, b = vec1, vec2
        v = np.cross(a, b)
        if np.sum(np.abs(v)) < .000001:
            return np.eye(3)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    
    def apply_global_trans(self, trans):
        for joints in self.all_joints:
            joints.pose += trans

    def get_all_rots(self):
        rots = []
        for i in range(21):
            rots.append(self.all_joints[i].local_frame)
            #see = self.all_joints[i].local_frame.as_euler("xyz", degrees=True)
            #print(see)
        return rots
    
    def get_all_rots_vecs(self):
        #upward vec:[0, 1, 0]; forward vec: [0, 0, 1]
        rots = []
        origin_rot_vecs = np.array([[0, 1, 0], [0, 0, 1]])
        for i in range(21):
            target_vecs = self.all_joints[i].local_frame.apply(origin_rot_vecs)
            #target_vecs = target_vecs.flatten()
            rots.append(target_vecs)
        return rots

    
    def get_parent_17_rots(self):
        rots = []
        # 0 1 2 3 [4] 5 6 7 [8] 9 T T1 12 13 14 15 16 17 18 [19] [20]
        pickOrder = [0,1,2,3,5,6,7,9,10,11,12,13,14,15,16,17,18]
        for i in range(17):
            rots.append(self.all_joints[pickOrder[i]].local_frame)
        return rots
    
    def get_all_poses(self):
        poses = []
        for i in range(21):
            poses.append(self.all_joints[i].pose)
        return np.array(poses)

    def construct_rot(self):
        
        def operate_construct_joint(current_joint):
            rot_T2Init = None
            source_vec = []
            desc_vec = []
            if current_joint.num > 0:
                for child in current_joint.children:
                    desc_vec.append(singleJoint.bone_vec_norm(child))
                    source_vec.append(current_joint.child_Tpose_norm[child.num])
                if len(source_vec) == 1:
                    rot_T2Init = self.rotation_matrix_from_vectors(source_vec[0], desc_vec[0])
                    rot_T2Init = Rot.from_matrix(rot_T2Init)
                elif len(source_vec) > 1:
                    try:
                        rot_T2Init = Rot.align_vectors(np.array(desc_vec), np.array(source_vec))[0]
                    except np.linalg.LinAlgError:
                        print("error here")
                        print(current_joint.num)
                        print(np.array(desc_vec))
                        print(np.array(source_vec))
                        print(current_joint.child_Tpose_norm)
                        exit()
            else:
                rot_T2Init = self.init_global_rot
            
            if not rot_T2Init is None:
                current_joint.set_local_frame(rot_T2Init)
            for child in current_joint.children:
                operate_construct_joint(child)

        operate_construct_joint(self.all_joints[0])

    
    def forward_kinematics_21Joints(self, applyRots):
        # applyRots  21*R
        
        def operate_FK(current_joint):
            current_joint.apply_rot(applyRots[current_joint.num])
            for child in current_joint.children:
                operate_FK(child)
        
        operate_FK(self.all_joints[0])
    
    def forward_kinematics_21Joints_vecs(self, applyRots):
        #applyRots ndarray->shape=(21,6)
        def operate_FK(current_joint):
            rotVec_y = applyRots[current_joint.num][:3]
            rotVec_z = applyRots[current_joint.num][3:]
            rotVec_x = np.cross(rotVec_y, rotVec_z)
            rotVec_x = rotVec_x / np.linalg.norm(rotVec_x)
            rotVec = np.concatenate((rotVec_x, rotVec_y, rotVec_z)).reshape((3,3))
            
            currentRotation = Rot.from_matrix(rotVec).inv()
            current_joint.apply_rot(currentRotation)
            
            for child in current_joint.children:
                operate_FK(child)
        
        operate_FK(self.all_joints[0])
    
    def forward_kinematics_Legs_vecs(self, applyRots):
        #applyRots ndarray->shape=(9,6) (8, 6)
        legJointsNo = [0, 13, 14, 15, 19, 16, 17, 18, 20]
        rotsMap = {0:0, 13:1, 14:2, 15:3, 16:4, 17:5, 18:6, 19:7, 20:8}
        #legJointsNo = [13, 14, 15, 19, 16, 17, 18, 20]
        #rotsMap = { 13:0, 14:1, 15:2, 16:3, 17:4, 18:5, 19:6, 20:7}
        def operate_FK(current_joint):
            if current_joint.num in legJointsNo:
                rotVec_y = applyRots[rotsMap[current_joint.num]][:3]
                rotVec_z = applyRots[rotsMap[current_joint.num]][3:]
                rotVec_x = np.cross(rotVec_y, rotVec_z)
                rotVec_x = rotVec_x / np.linalg.norm(rotVec_x)
                rotVec = np.concatenate((rotVec_x, rotVec_y, rotVec_z)).reshape((3,3))
                currentRotation = Rot.from_matrix(rotVec).inv()

                current_joint.apply_rot(currentRotation)
            for child in current_joint.children:
                operate_FK(child)
        
        operate_FK(self.all_joints[0])
        
    def forward_kinematics_UpperBody_vecs(self, applyRots):
        #applyRots ndarray->shape=(12,6)
        UpperJoints = [j for j in range(1, 13)]
        def operate_FK(current_joint):
            if current_joint.num in UpperJoints:
                rotVec_y = applyRots[current_joint.num - 1][:3]
                rotVec_z = applyRots[current_joint.num - 1][3:]
                rotVec_x = np.cross(rotVec_y, rotVec_z)
                rotVec_x = rotVec_x / np.linalg.norm(rotVec_x)
                rotVec = np.concatenate((rotVec_x, rotVec_y, rotVec_z)).reshape((3,3))
                currentRotation = Rot.from_matrix(rotVec).inv()

                current_joint.apply_rot(currentRotation)
            for child in current_joint.children:
                operate_FK(child)
        
        operate_FK(self.all_joints[0])
    
    def forward_kinematics_parentsOnly_17Joints(self, applyRots):
        # applyRots  17*R   Note: J_4=J_3  J7=J8   J19=J15  J20=J18 in orignal No.
        #           0 1 2 3 4 5 6 7 8 9 T 1 12 13 14 15 16 17 18 19 20
        newOrder = [0,1,2,3,3,4,5,6,6,7,8,9,10,11,12,13,14,15,16,13,16]
        temp = []
        for i in range(21):
            temp.append(applyRots[newOrder[i]])
        applyRots = temp
        
        def operate_FK(current_joint):
            current_joint.apply_rot(applyRots[current_joint.num])
            for child in current_joint.children:
                operate_FK(child)
        
        operate_FK(self.all_joints[0])
    
    def fk_rootY_test(self, applyRot):
        targetRot = Rot.from_euler("y", applyRot, degrees=True)
        self.all_joints[0].apply_rot(targetRot)
    
    def generate_standard_Tpose_from_this_bone_length(self):
        bone_length = []
        for connection in self.joints_connection:
            start, end = connection
            diffHere = self.all_joints[end].pose - self.all_joints[start].pose
            diffHere = np.linalg.norm(diffHere)
            
            if end in [5,6,7,8,9,10,11,12] and diffHere < 0.1:
                diffHere = 0.21  #re-construct bones if arms are losing
            bone_length.append(diffHere)
                
            
        bone_length = np.array(bone_length)
        bodyWhole_T = np.array([[.0, .0, .0] for k in range(21) ])
        for k, connection in enumerate(self.joints_connection):
            start, end = connection
            bodyWhole_T[end] = bodyWhole_T[start] + np.array(self.joints_Tpose_norm[k]) * bone_length[k]
        return bodyWhole_T
        
            
