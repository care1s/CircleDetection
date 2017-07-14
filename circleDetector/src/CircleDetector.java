import java.util.ArrayList;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

public class CircleDetector {
    public static void main(String... args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String address = "./test2.jpg";
        String code = image2code(address);
        System.out.println(code);
    }

    public static Mat edge_detect(Mat image_gray) {
        Mat image_gray_top = new Mat();
        Mat image_gray_right = new Mat();
        Mat image_gray_left = new Mat();
        Mat image_gray_bottom = new Mat();

        // 初步去噪，使用高斯，自定义高斯核进行处理，看效果
        try {

            Mat filter_kernel_bottom = new Mat(3, 3, CvType.CV_32F) {{
                put(0, 0, -1);
                put(0, 1, -2);
                put(0, 2, -1);

                put(1, 0, 0);
                put(1, 1, 0);
                put(1, 2, 0);

                put(2, 0, 1);
                put(2, 1, 2);
                put(2, 2, 1);

            }};

            Mat filter_kernel_left = new Mat(3, 3, CvType.CV_32F) {{
                put(0, 0, 1);
                put(0, 1, 0);
                put(0, 2, -1);

                put(1, 0, 2);
                put(1, 1, 0);
                put(1, 2, -2);

                put(2, 0, 1);
                put(2, 1, 0);
                put(2, 2, -1);

            }};

            Mat filter_kernel_top = new Mat(3, 3, CvType.CV_32F) {{
                put(0, 0, 1);
                put(0, 1, 2);
                put(0, 2, 1);

                put(1, 0, 0);
                put(1, 1, 0);
                put(1, 2, 0);

                put(2, 0, -1);
                put(2, 1, -2);
                put(2, 2, -1);

            }};

            Mat filter_kernel_right = new Mat(3, 3, CvType.CV_32F) {{
                put(0, 0, -1);
                put(0, 1, 0);
                put(0, 2, 1);

                put(1, 0, -2);
                put(1, 1, 0);
                put(1, 2, 2);

                put(2, 0, -1);
                put(2, 1, 0);
                put(2, 2, 1);

            }};

            final int filter_ddepth = -1;

            Imgproc.filter2D(image_gray,image_gray_top,filter_ddepth,filter_kernel_top);
            Imgproc.filter2D(image_gray,image_gray_right,filter_ddepth,filter_kernel_right);
            Imgproc.filter2D(image_gray,image_gray_bottom,filter_ddepth,filter_kernel_bottom);
            Imgproc.filter2D(image_gray,image_gray_left,filter_ddepth,filter_kernel_left);

            Imgcodecs.imwrite("test_new_2.jpg",image_gray_top);
        }catch(Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
        Mat image_edge = new Mat(image_gray.size(),CvType.CV_8U);
        // make the average of four direction into the edge!
        double threshold_around = 200.0;
        for(int i = 0 ; i < image_edge.width() ; i ++) {
            for (int k = 0; k < image_edge.height(); k++) {
                if ((image_gray_bottom.get(k, i)[0] > threshold_around) || (image_gray_left.get(k, i)[0] > threshold_around) ||
                        (image_gray_top.get(k, i)[0] > threshold_around) || (image_gray_right.get(k, i)[0] > threshold_around)) {
                    image_edge.put(k, i, 255);
                } else {
                    image_edge.put(k, i, 0);
                }
            }
        }
        return image_edge;
    }


    public static String image2code(String address) {
        String circle_code = "1";
        Mat image_original= Imgcodecs.imread(address);
        Mat image_gray = new Mat(image_original.size(),CvType.CV_8U,new Scalar(0));
        Imgproc.cvtColor(image_original, image_gray, Imgproc.COLOR_BGR2GRAY);

        Mat image_edge = edge_detect(image_gray);


        //do you need dilate or erode?
        Mat image_edge_dilate = new Mat();
        Mat ele_dilate = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,new Size(3,3));
        Imgproc.dilate(image_edge,image_edge_dilate,ele_dilate);

        Mat image_edge_dilate_copy = new Mat();
        image_edge_dilate.copyTo(image_edge_dilate_copy);
        // we face the big problem that the original image changes in the findContours

        // find the contours of original image
        ArrayList<MatOfPoint> target_mask_contours = new ArrayList<MatOfPoint>();		//find the contours of images
        Mat hierarchy_target_mask = new Mat();
        hierarchy_target_mask.convertTo(hierarchy_target_mask, CvType.CV_8U);
        Imgproc.findContours(image_edge_dilate, target_mask_contours,hierarchy_target_mask, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int max_index = 0;
        double max_area = 0;
        for (int i = 0; i < (int)target_mask_contours.size(); i++)
        {
            double g_dConArea = Imgproc.contourArea(target_mask_contours.get(i));
            //System.out.println(g_dConArea);
            if (g_dConArea >= max_area) {
                max_area = g_dConArea;
                max_index = i;
            }
        }
        Mat target_mask = new Mat(image_edge_dilate.size(), CvType.CV_8U,new Scalar(0));
        Imgproc.drawContours(target_mask, target_mask_contours, max_index,new Scalar(255), -1);	//ÖÐŒäÇøÓòµÄmask

        Mat target_mask_all = new Mat(image_edge_dilate.size(), CvType.CV_8U,new Scalar(0));
        Imgproc.drawContours(target_mask_all, target_mask_contours, -1,new Scalar(255), -1);	//ÖÐŒäÇøÓòµÄmask

        // 矫正外界矩形
        MatOfPoint2f out_mp2f =new MatOfPoint2f( target_mask_contours.get(max_index).toArray());
        RotatedRect outRect = Imgproc.minAreaRect(out_mp2f);
        Point[] P_outRect = new Point[4];
        outRect.points(P_outRect);

        // the angle is [90 -90]
        double angle_rect = outRect.angle;
        Mat rotation_mid = new Mat(image_gray.size(),CvType.CV_8U);

        if(angle_rect >=0.0) {
            Mat M = Imgproc.getRotationMatrix2D(new Point(image_edge_dilate.rows()/2,image_edge_dilate.cols()/2), -angle_rect, 1.0);
            Imgproc.warpAffine(image_gray, rotation_mid, M, image_edge_dilate.size());
        }else {
            Mat M = Imgproc.getRotationMatrix2D(new Point(image_edge_dilate.rows()/2,image_edge_dilate.cols()/2), angle_rect, 1.0);
            Imgproc.warpAffine(image_gray, rotation_mid, M, image_edge_dilate.size());
        }

        ArrayList<MatOfPoint> target_mask_contours_to_find_circle = new ArrayList<MatOfPoint>();		//find the contours of images
        Mat hierarchy_target_mask_to_find_circle = new Mat();
        hierarchy_target_mask_to_find_circle.convertTo(hierarchy_target_mask_to_find_circle, CvType.CV_8U);
        Imgproc.findContours(target_mask_all, target_mask_contours_to_find_circle,hierarchy_target_mask_to_find_circle, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int max_index_to_find_circle = 0;
        double max_area_to_find_circle = 0;
        for (int i = 0; i < (int)target_mask_contours_to_find_circle.size(); i++)
        {
            double g_dConArea = Imgproc.contourArea(target_mask_contours_to_find_circle.get(i));
            //System.out.println(g_dConArea);
            if (g_dConArea >= max_area_to_find_circle) {
                max_area_to_find_circle = g_dConArea;
                max_index_to_find_circle = i;
            }
        }

        // 寻找四个圆的圆心
        int radius = 0;     // here the radius is not right?
        ArrayList<Point> points_circle = new ArrayList<Point>();
        for (int i = 0 ; i < (int)target_mask_contours_to_find_circle.size(); i ++) {
            if(i != max_index_to_find_circle) {
                double x_all = 0;
                double y_all = 0;
                for (int k = 0; k < target_mask_contours_to_find_circle.get(i).toArray().length; k++) {
                    x_all = x_all + target_mask_contours_to_find_circle.get(i).toArray()[k].x;
                    y_all = y_all + target_mask_contours_to_find_circle.get(i).toArray()[k].y;
                }
                int pre_x = (int)x_all / target_mask_contours_to_find_circle.get(i).toArray().length;
                int pre_y = (int)y_all / target_mask_contours_to_find_circle.get(i).toArray().length;

                Point xy = new Point(pre_x, pre_y);
                points_circle.add(xy);

            }
        }

        // the angle 2 is wrong. we need to fixed it ?
        double rot_angle2 = 0.0;
        for(int i = 0 ;i < points_circle.size(); i ++) {
            if(image_edge_dilate.get((int)points_circle.get(i).x, (int)points_circle.get(i).y - 3)[0] > 150) {
                if (points_circle.get(i).y > image_edge_dilate.rows()/2 && points_circle.get(i).x > image_edge_dilate.cols()/2) {
                    rot_angle2 = 0;
                }
                else if (points_circle.get(i).y < image_edge_dilate.rows()/2 && points_circle.get(i).x > image_edge_dilate.cols()/2) {
                    rot_angle2 = 0;
                }
                else if (points_circle.get(i).y < image_edge_dilate.rows()/2 && points_circle.get(i).x < image_edge_dilate.cols()/2) {
                    rot_angle2 = 0;
                }else {
                    rot_angle2 = 0;
                }
            }
        }

        Mat rotation_final = new Mat(image_edge_dilate.size(),CvType.CV_8U);

        Mat M = Imgproc.getRotationMatrix2D(new Point(image_edge_dilate.rows()/2,image_edge_dilate.cols()/2), rot_angle2, 1.0);
        Imgproc.warpAffine(rotation_mid, rotation_final, M, image_edge_dilate.size());


        Mat image_edge_rotated = edge_detect(rotation_final);
        //do you need dilate or erode?
        Mat image_edge_dilate_rotated = new Mat();
        Imgproc.dilate(image_edge_rotated,image_edge_dilate_rotated,ele_dilate);

        ArrayList<MatOfPoint> target_mask_contours_justified = new ArrayList<MatOfPoint>();		//find the contours of images
        Mat hierarchy_target_mask_justified = new Mat();
        hierarchy_target_mask_justified.convertTo(hierarchy_target_mask_justified, CvType.CV_8U);
        Imgproc.findContours(image_edge_dilate_rotated, target_mask_contours_justified,hierarchy_target_mask_justified, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        int max_index_justified = 0;
        double max_area_justified = 0;
        for (int i = 0; i < (int)target_mask_contours_justified.size(); i++)
        {
            double g_dConArea = Imgproc.contourArea(target_mask_contours_justified.get(i));
            //System.out.println(g_dConArea);
            if (g_dConArea >= max_area_justified) {
                max_area_justified = g_dConArea;
                max_index_justified = i;
            }
        }
        Mat target_mask_justified = new Mat(rotation_final.size(), CvType.CV_8U,new Scalar(0));
        Imgproc.drawContours(target_mask_justified, target_mask_contours_justified, max_index_justified,new Scalar(255), -1);	//ÖÐŒäÇøÓòµÄmask

        MatOfPoint2f out_mp2f_justified =new MatOfPoint2f( target_mask_contours_justified.get(max_index_justified).toArray());
        RotatedRect outRect_justified = Imgproc.minAreaRect(out_mp2f_justified);
        Point[] P_outRect_justified = new Point[4];
        outRect_justified.points(P_outRect_justified);

        int out_rect_height = 0;
        int out_rect_width = 0;
        int left_top_point_x = (int)P_outRect_justified[0].x;
        int left_top_point_y = (int)P_outRect_justified[0].y;
        for(int i = 1 ; i < 4 ; i ++) {
            out_rect_height += Math.abs(P_outRect_justified[0].y - P_outRect_justified[i].y);
            out_rect_width += Math.abs(P_outRect_justified[0].x - P_outRect_justified[i].x);

            // find the point of left top
            if(left_top_point_x + left_top_point_y > P_outRect_justified[i].x + P_outRect_justified[i].y) {
                left_top_point_x = (int)P_outRect_justified[i].x;
                left_top_point_y = (int)P_outRect_justified[i].y;
            }
        }
        out_rect_height = out_rect_height/2;
        out_rect_width = out_rect_width/2;

        Mat roi_target = new Mat(rotation_final,new Rect(left_top_point_x,
                left_top_point_y, out_rect_width, out_rect_height/10));

        // message above used to get the target of the images.
        // the key question is that how to make the point to the (x,y), k-means


        // 这里可能需要对前背景进行分离
	    double m = Core.mean(roi_target).val[0];
        Mat thre = new Mat(roi_target.size(),CvType.CV_8UC1,new Scalar(0));
        Imgproc.threshold(roi_target, thre, m, 255, Imgproc.THRESH_BINARY);
        Mat ele_erode_final = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(5,5));
        Mat thre_erode = new Mat();
        Imgproc.erode(thre,thre_erode,ele_erode_final);

        Mat ele_dilate_final = Imgproc.getStructuringElement(Imgproc.MORPH_CROSS,new Size(3,3));
        Mat thre_dst = new Mat();
        Imgproc.dilate(thre_erode,thre_dst,ele_dilate_final); //这张图是可行的。但是有问题的。

        Imgcodecs.imwrite("test_new.jpg",thre_dst);
// 因为kmeans算法很容易失败，因此这里就不使用kmeans算法了。这里只是使用最简单的阈值分割 这里发现我们对处理后的图像进行边缘轮廓提取时，提不出了轮廓，为什么？

        ArrayList<MatOfPoint> circle_contours = new ArrayList<MatOfPoint>();
        Mat hierarchy_circle_find = new Mat();
        hierarchy_circle_find.convertTo(hierarchy_circle_find,CvType.CV_8U);
        Imgproc.findContours(thre_dst,circle_contours,hierarchy_circle_find,Imgproc.RETR_LIST,Imgproc.CHAIN_APPROX_NONE);

        Mat test_image_dst = new Mat(thre_dst.size(),CvType.CV_8U,new Scalar(0));
        //Imgproc.drawContours(test_image_dst,circle_contours,-1,new Scalar(255),-1);

//        for(int i = 0 ; i < circle_contours.size() ; i ++) {
//            double g_dConArea = Imgproc.contourArea(circle_contours.get(i));
//            System.out.println(g_dConArea);
//        }
        ArrayList<Point> circle_point = new ArrayList<Point>();
	    	for (int i = circle_contours.size()-1; i >= 0; i--) {
	    		double g_dConArea = Imgproc.contourArea(circle_contours.get(i));
	    		if (g_dConArea > 15 && g_dConArea < 70){
	    			double x_all = 0;
	    			double y_all = 0;
	    			for (int k = 0; k < circle_contours.get(i).toArray().length; k++) {
	    				x_all = x_all + circle_contours.get(i).toArray()[k].x;
	    				y_all = y_all + circle_contours.get(i).toArray()[k].y;
	    			}

	    			int pre_x = (int)x_all / circle_contours.get(i).toArray().length;
	    			int pre_y = (int)y_all / circle_contours.get(i).toArray().length;

	    			Point xy = new Point(pre_x, pre_y);
	    			circle_point.add(xy);
	    		}
	    	}

	    	// re-arrange the point
	    	ArrayList<Point> circle_point_new = new ArrayList<Point>();
	    	while( circle_point.size() != 0) {
	    		int min = 0;
	    		for (int i = 1; i < circle_point.size(); i ++) {
	    			if(circle_point.get(min).x > circle_point.get(i).x
	    					&& Math.abs(circle_point.get(min).y-circle_point.get(i).y) <3){
	    				min = i;
	    			}
	    		}
	    		circle_point_new.add(circle_point.get(min));
	    		circle_point.remove(min);
	    	}

	    	for(int i = 0 ; i < circle_point_new.size(); i ++) {
	    	    System.out.println(circle_point_new.get(i).x + "   " + circle_point_new.get(i).y);

            }

	    	if (circle_point_new.size() > 1) {
	    		try {
		    	int dot_marge = 13;//(int)(circle_point_new.get(1).x-circle_point_new.get(0).x);

		    	for (int i = 1 ; i < 16 ; i ++) {
		    		int side_distance = (int)(Math.abs(circle_point_new.get(i).x - circle_point_new.get(i - 1).x) / dot_marge)-1;
		    		for (int k = 0 ; k < side_distance;k++) {
		    			circle_code +="0";
		    		}
		    		circle_code +="1";
		    	}
		    	// Á¬ÐøÈýžöµãÎªÖÕÖ¹·û
//		    	int i = 1;
//		    	while(circle_code.contains("111") != true) {
//		    		int side_distance = (int)((circle_point_new.get(i).x - circle_point_new.get(i - 1).x) / dot_marge) - 1;
//		    		for (int k = 0 ; k < side_distance;k++) {
//		    			circle_code +="0";
//		    		}
//		    		circle_code +="1";
//		    	}
	    		}catch(Exception e) {
	    			circle_code = "Exception out";
	    		}
	    	}else {
	    		circle_code="DETECT WRONG!";
	    	}
        return circle_code;
    }
    public static void pixelLess(Mat image, Mat new_image) {

        int count = 20;
        for(int i = 0 ; i < image.rows() ; i ++) {
            for(int k = 0 ; k < image.cols() ; k ++) {
                new_image.put(i,k, ((int)image.get(i,k)[0]/count)*count);
            }
        }

    }
    public static void find_two_large_pixel(Mat image, int[] a) {
        int[] hist = new int[255];

        for(int x = 0 ; x < hist.length ; x ++) {
            hist[x] = 0;
        }

        for(int i = 0 ; i < image.rows() ; i ++) {
            for(int k = 0 ; k < image.cols() ; k ++) {
                hist[(int)image.get(i,k)[0]] ++;
            }
        }

        int max = 0;
        int second = 0;
        for(int i = 0 ; i < 255 ; i ++) {
            if(hist[max] < hist[i]) {
                second = max;
                max = i;
            }
        }
        a[0] = max;
        a[1] = second;
    }

    public static void gray_myself(Mat image, Mat new_image) {

        for( int i = 0 ; i < image.rows() ; i ++) {
            for( int k = 0 ; k < image.cols() ; k ++) {
                new_image.put(i,k,(int)image.get(i,k)[0]*0.114 + image.get(i,k)[1]*0.587 + image.get(i,k)[2]*0.299);
            }
        }

    }

}