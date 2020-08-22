PATH_TO_FROZEN_GRAPH = '/home/ubuntu/cocacola_201904/coke_dataset/models/research/object_detection/legacy/models/train/'+ '/frozen_inference_graph.pb'
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/ubuntu/cocacola_201904/coke_dataset/training', 'cocacola_label.pbtxt') 
# label box is binary bbox to put into training
NUM_CLASSES = 2

graph = tf.Graph()

with graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    
    return np.array(image.getdata())\
    .reshape((im_height, im_width, 3))\
    .astype(np.uint8)