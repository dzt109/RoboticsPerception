## Project: 3D Perception Pick & Place
## Deepak Trivedi, 11/23/2017


[//]: # (Image References)

[image1]: ./misc_images/train_svm.PNG
[image2]: ./misc_images/world2a.PNG
[image3]: ./misc_images/world3.PNG


---

This writeup describes  the implementation of an RGBD camera based perception for a pick and place robot. Perception is based on a Support Vector Machine (SVM) based classification model. Three different 'worlds' were trained and tested for this exercise. These worlds contained perception tasks of increasing complexity. 

Following is an overview of the steps invloved in this process: 

### Steps involved in the pick and place perception exercise
1. Extract features and train an SVM model on new objects. 
2. Write a ROS node and subscribe to `/pr2/world/points` topic. This topic contains noisy point cloud data that you must work with.
3. Use filtering and RANSAC plane fitting to isolate the objects of interest from the rest of the scene.
4. Apply Euclidean clustering to create separate clusters for individual items.
5. Perform object recognition on these objects and assign them labels (markers in RViz).
6. Calculate the centroid (average in x, y and z) of the set of points belonging to that each object.
7. Create ROS messages containing the details of each object (name, pick_pose, etc.) and write these messages out to `.yaml` files, one for each of the 3 scenarios (`test1-3.world` in `/pr2_robot/worlds/`).  
8. Submit a link to your GitHub repo for the project or the Python code for your perception pipeline and your output `.yaml` files (3 `.yaml` files, one for each test world).  You must have correctly identified 100% of objects from `pick_list_1.yaml` for `test1.world`, 80% of items from `pick_list_2.yaml` for `test2.world` and 75% of items from `pick_list_3.yaml` in `test3.world`.


The PR2 has been outfitted with an RGB-D sensor much like the one you used in previous exercises. This sensor however is a bit noisy, much like real sensors.

Given the cluttered tabletop scenario, you must implement a perception pipeline using your work from Exercises 1,2 and 3 to identify target objects from a so-called “Pick-List” in that particular order, pick up those objects and place them in corresponding dropboxes.

## Installation
For this setup, catkin_ws is the name of active ROS Workspace, if your workspace name is different, change the commands accordingly
If you do not have an active ROS workspace, you can create one by:

```sh
$ mkdir -p ~/catkin_ws/src
$ cd ~/catkin_ws/
$ catkin_make
```

Now that you have a workspace, clone or download this repo into the src directory of your workspace:
```sh
$ cd ~/catkin_ws/src
$ git clone https://github.com/udacity/RoboND-Perception-Project.git
```
### Note: If you have the Kinematics Pick and Place project in the same ROS Workspace as this project, please remove the 'gazebo_grasp_plugin' directory from the `RoboND-Perception-Project/` directory otherwise ignore this note. 

Now install missing dependencies using rosdep install:
```sh
$ cd ~/catkin_ws
$ rosdep install --from-paths src --ignore-src --rosdistro=kinetic -y
```
Build the project:
```sh
$ cd ~/catkin_ws
$ catkin_make
```
Add following to your .bashrc file
```
export GAZEBO_MODEL_PATH=~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/models:$GAZEBO_MODEL_PATH
```

If you haven’t already, following line can be added to your .bashrc to auto-source all new terminals
```
source ~/catkin_ws/devel/setup.bash
```

To run the demo:
```sh
$ cd ~/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts
$ chmod u+x pr2_safe_spawner.sh
$ ./pr2_safe_spawner.sh
```
![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)



Once Gazebo is up and running, make sure you see following in the gazebo world:
- Robot

- Table arrangement

- Three target objects on the table

- Dropboxes on either sides of the robot


If any of these items are missing, please report as an issue on [the waffle board](https://waffle.io/udacity/robotics-nanodegree-issues).

In your RViz window, you should see the robot and a partial collision map displayed:

![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Proceed through the demo by pressing the ‘Next’ button on the RViz window when a prompt appears in your active terminal

The demo ends when the robot has successfully picked and placed all objects into respective dropboxes (though sometimes the robot gets excited and throws objects across the room!)

Close all active terminal windows using **ctrl+c** before restarting the demo.

You can launch the project scenario like this:
```sh
$ roslaunch pr2_robot pick_place_project.launch
```

## Capturing Features and SVM training

For generating training and test data, each of the objects in the world was spawned in 15 random orientations, and the resulting point clouds captured by the RGBD camera were used for training the model. 

A Support Vector Machine (SVM) is the model of choice for capturing features of objects for perception. The features are:

1. Normalized histograms of the color for each point in the cloud
2. Surface normal vectors for each point in the point cloud 

These features are generated by the code `features.py`.  Both for HSV histograms and surface normal histograms, the number of bins used was 32. 
 
```python 

    # TODO: Compute histograms
    nbins=32
    bins_range=(0, 256)
    h_hist = np.histogram(channel_1_vals, bins=nbins, range=bins_range)
    s_hist = np.histogram(channel_2_vals, bins=nbins, range=bins_range)
    v_hist = np.histogram(channel_3_vals, bins=nbins, range=bins_range)
    # TODO: Concatenate and normalize the histograms
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0])).astype(np.float64)
    # Normalize the result
    normed_features = hist_features / np.sum(hist_features)
``` 
 
It was found that when the color is expressed in HSV format, better results are obtained regardless of the lighting conditions. 


An example of the confusion matrix is shown below:
 
![alt text][image1] 

## Pipeline for Perception

The code for perception is available here. 

### Filtering and RANSAC plane fitting

This is implemented via the following chunk 

```python 

    # TODO: Convert ROS msg to PCL data
    origcloud = ros_to_pcl(pcl_msg)

    #Filter noise

    noise_filter = origcloud.make_statistical_outlier_filter()
    noise_filter.set_mean_k(50)
    noise_filter.set_std_dev_mul_thresh(0.5)
    cloud = noise_filter.filter()

    # TODO: Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.002
    vox.set_leaf_size(LEAF_SIZE,LEAF_SIZE,LEAF_SIZE)
    cloud_filtered = vox.filter()


    # TODO: PassThrough Filter

    cloud_filtered = three_axis_passthrough_filter(cloud_filtered,xmin=0.6,xmax=1.3,ymin=0.3,ymax=1.0,zmin=-0.5,zmax=0.5)


    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

```

Here, the function `three_axis_passthrough_filter` was defined as 

```python 

	#1-axis passthrough filter
	def passthrough_filter(point_cloud, name_axis, min_axis, max_axis):
	  pass_filter = point_cloud.make_passthrough_filter()
	  pass_filter.set_filter_field_name(name_axis)
	  pass_filter.set_filter_limits(min_axis, max_axis)
	  return pass_filter.filter()
	
	#3-axis passthrough filter
	def three_axis_passthrough_filter(cloud,xmin,xmax,ymin,ymax,zmin,zmax):
		  cloud_z = passthrough_filter(point_cloud = cloud, 
			                                 name_axis = 'z', min_axis = xmin, max_axis = xmax)
	
		  cloud_zx = passthrough_filter(point_cloud = cloud_z, 
			                                 name_axis = 'x', min_axis = ymin, max_axis = ymax)
	
		  cloud_zxy = passthrough_filter(point_cloud = cloud_zx, 
			                                 name_axis = 'y', min_axis = zmin, max_axis = zmax)
	
		  return cloud_zxy
```
### Clustering for segmentation

This is implemented as follows :

```python

	# TODO: Extract inliers and outliers
    inliers, coefficients = seg.segment()
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
    # Set tolerances for distance threshold 
    # as well as minimum and maximum cluster size (in points)
   
    ec.set_ClusterTolerance(0.01)
    ec.set_MinClusterSize(200)
    ec.set_MaxClusterSize(15000)
    # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))

    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                         rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

``` 

### Object recognition implementation.

Following code does object recognition implementation for each of the clusters identified. 

```python

    # Classify the clusters! (loop through each detected cluster one at a time)
    # Classify the clusters!
    detected_objects_labels = []
    detected_objects = []

    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = extracted_outliers.extract(pts_list)
        # TODO: convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # TODO: complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
```

### Pick and Place Setup

The following code sets the simulator up for pick and place, and writes YAML files. 

```python

    # TODO: Initialize variables
    test_scene = Int32()
    test_scene.data = TEST_SCENE
    OUTPUT_FILENAME = "output_" + str(TEST_SCENE) + '.yaml'
    output = []

    # TODO: Get/Read parameters
    pick_list = rospy.get_param('/object_list')
    dropbox_list = rospy.get_param('/dropbox')


    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # TODO: Loop through the pick list
    name = String()
    arm = String()
    pick_pose = Pose()
    place_pose =  Pose()
    for object in object_list:

        # TODO: Get the PointCloud for a given object and obtain it's centroid
	
	points = ros_to_pcl(object.cloud).to_array()
	x, y, z = np.mean(points, axis = 0)[:3]
	pick_pose.position.x = np.asscalar(x) 
        pick_pose.position.y = np.asscalar(y)
        pick_pose.position.z = np.asscalar(z)
        # TODO: Create 'place_pose' for the object
        
	for pickobject in pick_list:
		if pickobject['name'] == str(object.label):
		   group = pickobject['group']
	           name.data = str(object.label)
		   break

        for box in dropbox_list:
        	if box['group'] == group:
		        x, y, z = box['position']
        		place_pose.position.x = np.float(x) 
        		place_pose.position.y = np.float(y)
        		place_pose.position.z = np.float(z)        
        		arm.data = box['name']
        		break



        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        yaml_dict = make_yaml_dict(test_scene, arm, name, pick_pose, place_pose)
	output.append(yaml_dict)

        # Wait for 'pick_place_routine' service to come up
        #rospy.wait_for_service('pick_place_routine')

        #try:
        #   pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your message variables to be sent as a service request
        #    resp = pick_place_routine(test_scene, name, arm, pick_pose, place_pose)

        #    print ("Response: ",resp.success)

        #except rospy.ServiceException, e:
        #    print "Service call failed: %s"%e

    # TODO: Output your request parameters into output yaml file
    send_to_yaml(OUTPUT_FILENAME, output)
    print "Sent to file"

``` 

## Results

For all three tabletop setups (`test*.world`), first, SVM models were trained, and then object recognition was performed using the RGBD camera in the simulated environment. Then, valid `PickPlace` request messages were output in `.yaml` format.


### Output YAML files
Following are output `.yaml` files, one for each of the three tabletop setups. 

[output_1.yaml](./pr2_robot/scripts/output_1.yaml)

[output_2.yaml](./pr2_robot/scripts/output_2.yaml)

[output_3.yaml](./pr2_robot/scripts/output_3.yaml)

### Example of perception

Following examples of labels  applied after perception and classification for different tabletop setups. 

![alt text][image2] 


![alt text][image3]

### List of code files

[train_svm.py](./pr2_robot/scripts/train_svm.py) - Code for training SVM on RGBD images. 

[capture_features.py](./pr2_robot/scripts/capture_features.py) - Captures images in random orientations

[features.py](./pr2_robot/scripts/features.py) - Code for generating features from images.

[project_perception.py](./pr2_robot/scripts/project_perception.py) - Code for perception and writing `.yaml` files. 

### List of SVM models

[model1.sav](./pr2_robot/scripts/model1.sav) - SVM model for World 1

[model2.sav](./pr2_robot/scripts/model2.sav) - SVM model for World 2

[model3.sav](./pr2_robot/scripts/model3.sav) - SVM model for World 3

### Future work

The SVM models does a good job at identifying objects on the table. However, as the confusion matrix suggests, there is scope for improvement. Easy gains could be made by increasing the number of random samples used to train the model. 

I have not studied the effect of noise on the effectiveness of the perception pipeline. Increasing the amount of noise will help understand the limitations of the classifiers. 

The algorithm will get confused if the objects occult each other. Some thought will need to be given for alternative features required to handle this situation. 




