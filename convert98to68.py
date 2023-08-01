import os

def convert98to68pts(file_98pts, fw_path):
    infos_68 = []
    with open(file_98pts, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            list_info = line.strip().split(' ')
            points = list_info[0:196]
            info_68 = []
            for j in range(17):
                x = points[j*2*2+0]
                y = points[j*2*2+1]
                info_68.append(x)
                info_68.append(y)
            for j in range(33, 38):
                x = points[j*2+0]
                y = points[j*2+1]
                info_68.append(x)
                info_68.append(y)
            for j in range(42, 47):
                x = points[j*2+0]
                y = points[j*2+1]
                info_68.append(x)
                info_68.append(y)
            for j in range(51, 61):
                x = points[j*2+0]
                y = points[j*2+1]
                info_68.append(x)
                info_68.append(y)
            point_38_x = (float(points[60*2+0]) + float(points[62*2+0])) / 2.0
            point_38_y = (float(points[60*2+1]) + float(points[62*2+1])) / 2.0
            point_39_x = (float(points[62*2+0]) + float(points[64*2+0])) / 2.0
            point_39_y = (float(points[62*2+1]) + float(points[64*2+1])) / 2.0
            point_41_x = (float(points[64*2+0]) + float(points[66*2+0])) / 2.0
            point_41_y = (float(points[64*2+1]) + float(points[66*2+1])) / 2.0
            point_42_x = (float(points[60*2+0]) + float(points[66*2+0])) / 2.0
            point_42_y = (float(points[60*2+1]) + float(points[66*2+1])) / 2.0
            point_44_x = (float(points[68*2+0]) + float(points[70*2+0])) / 2.0
            point_44_y = (float(points[68*2+1]) + float(points[70*2+1])) / 2.0
            point_45_x = (float(points[70*2+0]) + float(points[72*2+0])) / 2.0
            point_45_y = (float(points[70*2+1]) + float(points[72*2+1])) / 2.0
            point_47_x = (float(points[72*2+0]) + float(points[74*2+0])) / 2.0
            point_47_y = (float(points[72*2+1]) + float(points[74*2+1])) / 2.0
            point_48_x = (float(points[68*2+0]) + float(points[74*2+0])) / 2.0
            point_48_y = (float(points[68*2+1]) + float(points[74*2+1])) / 2.0
            
            info_68.append(str(point_38_x))
            info_68.append(str(point_38_y))
            info_68.append(str(point_39_x)) 
            info_68.append(str(point_39_y))
            info_68.append(points[64*2+0])
            info_68.append(points[64*2+1])
            info_68.append(str(point_41_x))
            info_68.append(str(point_41_y))
            info_68.append(str(point_42_x))
            info_68.append(str(point_42_y))
            info_68.append(points[68*2+0])
            info_68.append(points[68*2+1])
            info_68.append(str(point_44_x))
            info_68.append(str(point_44_y))
            info_68.append(str(point_45_x))
            info_68.append(str(point_45_y))
            info_68.append(points[72*2+0])
            info_68.append(points[72*2+1])
            info_68.append(str(point_47_x))
            info_68.append(str(point_47_y))
            info_68.append(str(point_48_x))
            info_68.append(str(point_48_y))
            
            for j in range(76, 96):
                x = points[j*2+0]
                y = points[j*2+1]
                info_68.append(x)
                info_68.append(y)
            for j in range(len(list_info[196:])):
                info_68.append(list_info[196+j])
            infos_68.append(info_68)
    print(len(infos_68[0]))
    with open(fw_path, 'w') as fw:
        for i, line in enumerate(infos_68):
            
            for j in range(len(line)):
                fw.write(line[j]+' ')
            fw.write('\n')
    return 

if __name__ == '__main__':
    root_dir = '/content/drive/Shareddrives/FacialLandmark/input/WFLW/WFLW_annotations/'
                           
    file_98pts_train = os.path.join(root_dir + 'list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt')
    fw_train = os.path.join(root_dir + 'list_68pt_rect_attr_train.txt')
    convert98to68pts(file_98pts_train, fw_train)
                           
    file_98pts_test = os.path.join(root_dir + 'list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt')
    fw_test = os.path.join(root_dir + 'list_68pt_rect_attr_test.txt')
    convert98to68pts(file_98pts_test, fw_test)
                           
    file_98pts_test_blur = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_blur.txt')
    fw_test_blur = os.path.join(root_dir + 'list_68pt_test_blur.txt')
    convert98to68pts(file_98pts_test_blur, fw_test_blur) 
                           
    file_98pts_test_expression = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_expression.txt')
    fw_test_expression = os.path.join(root_dir + 'list_68pt_test_expression.txt')
    convert98to68pts(file_98pts_test_expression, fw_test_expression) 
                           
    file_98pts_test_illumination = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_illumination.txt')
    fw_test_illumination = os.path.join(root_dir + 'list_68pt_test_illumination.txt')
    convert98to68pts(file_98pts_test_illumination, fw_test_illumination) 
                           
    file_98pts_test_largepose = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_largepose.txt')
    fw_test_largepose = os.path.join(root_dir + 'list_68pt_test_largepose.txt')
    convert98to68pts(file_98pts_test_largepose, fw_test_largepose)
                           
    file_98pts_test_makeup = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_makeup.txt')
    fw_test_makeup = os.path.join(root_dir + 'list_68pt_test_makeup.txt')
    convert98to68pts(file_98pts_test_makeup, fw_test_makeup)
                           
    file_98pts_test_occlusion = os.path.join(root_dir + 'list_98pt_test/list_98pt_test_occlusion.txt')
    fw_test_occlusion = os.path.join(root_dir + 'list_68pt_test_occlusion.txt')
    convert98to68pts(file_98pts_test_occlusion, fw_test_occlusion)
                           