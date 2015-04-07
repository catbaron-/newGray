/*
process multicolor images
author: catbaron
date:2015-03-03 15:26:25
*/
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

const float PI = 3.1415;
const int MAX_REG = 5;
const int EDGE_WIDTH = 3;
int sumMat(Mat img, vector<Point> pts, int channel)
{
	int sum = 0;
	for (int i = 0; i < pts.size(); i++)
	{
		Point p = pts[i];
		sum = sum + (int)img.at<Vec3b>(p.y, p.x)[channel];
	}
	return sum;
}
class Edge
{
private:
	int width = EDGE_WIDTH;
	vector<vector<Point>> fpts;	//points on from-side of the edge
	vector<vector<Point>> tpts;	//points on to-side of the edge
	int rf, rt;	//this is an edge between regins of from-region and to-region
public:
	int getFromRegionNumber(){
		return rf;
	}
	int getToRegionNumber(){
		return rt;
	}
	/*void switchPoints(){
	for (int i = 0; i < pp.size(); i++)
	{
	Point p = pp[i][0];
	pp[i][0] = pp[i][1];
	pp[i][1] = p;
	}
	}*/
	void addPoints(vector<Point> pf, vector<Point> pt)
	{
		////去重
		//for (int i = 0; i < pp.size(); i++)
		//{
		//	Point f = pp[i][0];
		//	Point t = pp[i][1];
		//	if (f.x == pf.x && f.y == pf.y)
		//	{
		//		if (t.x == pt.x && t.y == pt.y)
		//			return;
		//	}
		//}
		//vector<Point> p;
		fpts.push_back(pf);
		tpts.push_back(pt);
	}
	void setRegions(int f, int t)
	{
		rf = f;
		rt = t;
	}
	vector<vector<Point>> getFromRegionPoints()
	{
		return fpts;
	}
	vector<vector<Point>> getToRegionPoints(){
		return tpts;
	}
};
class Region
{
private:
	vector<Point> pts;
	float JBG;
	//int reg;
	int bg;
	Point center;
	float J1, J2, J3;
	int max_channel;
	int rnum;
	float J(float a)
	{
		int cx = center.x;
		int cy = center.y;
		float AR = pts.size();
		float numerator = 0;
		float denominator = pow(AR, 1 + a / 2);

		////SUM{[(x - cx)^2 + (y - cy)^2]^(a/2)}////
		for (int i = 0; i < AR; i++)
		{
			Point p = pts[i];
			numerator = numerator + pow(pow(p.x - cx, 2) + pow(p.y - cy, 2), a / 2);
		}
		return numerator / denominator;
	}
	float JO(float a)
	{
		int S = pts.size();
		float R = pow(S / PI, 0.5);
		float jo = 2 * PI * pow(R, a) / pow(S, 1 + a / 2);
		return jo;
	}
public:

	void printInfo(int seq)
	{
		std::cout << "Region[" << seq << "]: " << endl;
		std::cout << "Center: " << center.x << ":" << center.y << endl;
		//		cout << "Js: " << J(1) << ":" << J(2) << ":" << J(3) << endl;
		std::cout << "JBG: " << JBG << endl;
		std::cout << "Js: " << J1 << ":" << J2 << ":" << J3 << endl;
		std::cout << "JOs: " << JO(1) << ":" << JO(2) << ":" << JO(3) << endl;
		std::cout << endl;
	}
	void setJs()
	{
		J1 = J(1);
		J2 = J(2);
		J3 = J(3);
	}
	float getJ(int a)
	{
		if (1 == a)
			return J1;
		else if (2 == a)
			return J2;
		else if (3 == a)
			return J3;
		return 0;
	}
	void setCenter()
	{
		int sumx = 0;
		int sumy = 0;
		int s = pts.size();
		for (int i = 0; i < s; i++)
		{
			sumx = sumx + pts[i].x;
			sumy = sumy + pts[i].y;
		}
		center.x = sumx / s;
		center.y = sumy / s;
	}
	Point getCenter()
	{
		return center;
	}

	float setJBG()
	{
		JBG = J(1) / JO(1) + J(2) / JO(2) + J(3) / JO(3);
		return JBG;
	}
	float getJBG()
	{
		return JBG;
	}

	void updateRegion(int i)
	{
		setCenter();
		setJs();
		setJBG();
		setRegionNum(i);
	}

	int getRegionNum()
	{
		return rnum;
	}
	void setRegionNum(int n)
	{
		rnum = n;
	}

	void addPoint(Point p)
	{
		pts.push_back(p);
	}
	vector<Point> getPoints()
	{
		return pts;
	}

	void setBackground()
	{
		bg = 1;
	}
	void setForeground()
	{
		bg = 0;
	}
	int isBackground()
	{
		return bg;
	}
	int isForekground()
	{
		return !bg;
	}
	void setMaxChannel(int c)
	{
		max_channel = c;
	}
	int getMaxChannel()
	{
		return max_channel;
	}
	int findMaxChannel(Mat img)
	{
		int b = 0, g = 0, r = 0;
		int max = 0;
		//vector<Mat> bgr;
		//split(img, bgr);
		b = sumMat(img, pts, 0);
		g = sumMat(img, pts, 1);
		r = sumMat(img, pts, 2);
		if (max < b)
			max = b;
		if (max < g)
			max = g;
		if (max < r)
			max = r;
		if (max == b)
		{
			setMaxChannel(0);
			return 0;
		}
		if (max == g)
		{
			setMaxChannel(1);
			return 1;
		}
		else
		{
			setMaxChannel(2);
			return 2;
		}
	}
	void updateRegionMap(int rn, Mat &res)
	{
		setRegionNum(rn);
		for (int i = 0; i < pts.size(); i++)
		{
			int x = pts[i].x;
			int y = pts[i].y;
			res.at<uchar>(y, x) = (uchar)rn;
		}
	}
};



void addToEdge(Point p, Mat region_map, vector<Edge> &edges)
{
	int x0 = p.x;
	int y0 = p.y;
	for (int dx = -1; dx < 2; dx++)
	{
		for (int dy = -1; dy < 2; dy++)
		{
			int ex0 = x0 + dx;
			int ey0 = y0 + dy;
			if (ex0 < 0)
				ex0 = 0;
			if (ex0 > region_map.cols - 1)
				ex0 = region_map.cols - 1;
			if (ey0 < 0)
				ey0 = 0;
			if (ey0 > region_map.rows - 1)
				ey0 = region_map.rows - 1;
			int t = (int)region_map.at<uchar>(ey0, ex0);
			int f = (int)region_map.at<uchar>(y0, x0);
			if (f != t)
			{
				//相邻两个点属于不同region
				//遍历Edge，把两个点添加到对应region间的edge去--0406更新，不止两个点，还包括这两个点（面向edge）身后2个点
				//如果没有此edge，则建立新的edge

				//遍历
				vector<Point> fp; //points of from-side
				int x1 = x0 - dx;
				int y1 = y0 - dy;
				int x2 = x0 - 2 * dx;
				int y2 = y0 - 2 * dy;

				x1 = (x1 < 0 || x1 > region_map.cols - 1) ? x0 : x1;
				x2 = (x2 < 0 || x2 > region_map.cols - 1) ? x1 : x2;
				y1 = (y1 < 0 || y1 > region_map.rows - 1) ? y0 : y1;
				y2 = (y2 < 0 || y2 > region_map.rows - 1) ? y1 : y2;

				fp.push_back(Point(x0, y0));
				fp.push_back(Point(x1, y1));
				fp.push_back(Point(x2, y2));

				vector<Point> tp; //points of to-side
				int ex1 = ex0 + dx;
				int ey1 = ey0 + dy;
				int ex2 = ex1 + dx;
				int ey2 = ey1 + dy;

				ex1 = (ex1 < 0 || ex1 > region_map.cols - 1) ? ex0 : ex1;
				ey1 = (ey1 < 0 || ey1 > region_map.rows - 1) ? ey0 : ey1;
				ex2 = (ex2 < 0 || ex2 > region_map.cols - 1) ? ex1 : ex2;
				ey2 = (ey2 < 0 || ey2 > region_map.rows - 1) ? ey1 : ey2;

				tp.push_back(Point(ex0, ey0));
				tp.push_back(Point(ex1, ey1));
				tp.push_back(Point(ex2, ey2));

				int find = 0;
				int fr, tr;//from-region-number and to-region-number
				for (int i = 0; i < edges.size(); i++)
				{
					fr = edges[i].getFromRegionNumber();
					tr = edges[i].getToRegionNumber();
					//找到edge，添加Points进去
					if (fr == f && tr == t)
					{
						find = 1;
						edges[i].addPoints(fp, tp);
					}
					if (fr == t && tr == f)
					{
						find = 1;
						edges[i].addPoints(tp, fp);
					}
				}
				//没有找到对应edge，新建edge
				if (!find)
				{
					Edge e;
					e.setRegions(f, t);
					e.addPoints(fp, tp);
					edges.push_back(e);
				}
				return;
			}
		}
	}
}
void segment(Mat &input_img, vector<Region> &rs, vector<Edge> &edges, Mat &region_map, Mat &region_show)
{
	Mat gray;
	cvtColor(input_img, gray, CV_RGB2GRAY);

	const int MAX_DIM_DES = input_img.cols * input_img.rows;	//number of pixel
	Mat des = Mat::zeros(MAX_DIM_DES, 3, CV_32FC1);	//descripter for segmentation
	Mat label;// result of kmeans

	//get descripter
	int cur = 0;
	for (int i = 0; i < input_img.rows; i++)
	{
		for (int j = 0; j < input_img.cols; j++)
		{
			int r = input_img.at<Vec3b>(i, j)[0];
			int g = input_img.at<Vec3b>(i, j)[1];
			int b = input_img.at<Vec3b>(i, j)[2];
			des.at<float>(cur, 0) = 1.0 * r;
			des.at<float>(cur, 1) = 1.0 *g;
			des.at<float>(cur, 2) = 1.0 *b;
			cur++;
		}
	}
	//get descripter done

	//segment by kmeans
	kmeans(des, MAX_REG, label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 20, 1.0), 10, KMEANS_PP_CENTERS);
	//segment done

	//result of segment as a map
	for (int i = 0; i < label.rows; i++)
	{
		int r = i / input_img.cols;
		int c = i % input_img.cols;
		region_map.at<uchar>(r, c) = ((uchar)label.at<int>(i));
		region_show.at<uchar>(r, c) = (50 * ((uchar)label.at<int>(i))) % 255;
	}
	//draw map done

	//get regions
	for (int i = 0; i < MAX_REG; i++)
	{
		Region r;
		r.setRegionNum(i);
		rs.push_back(r);
	}
	for (int i = 0; i < region_map.rows; i++)
	{
		for (int j = 0; j < region_map.cols; j++)
		{
			Point p(j, i);
			int l = (int)region_map.at<uchar>(i, j);
			rs[l].addPoint(p);
			//自动判断是否是edge，是的话就添加到edges里面
			addToEdge(p, region_map, edges);
		}
	}
	//get regions done
}

int findBackground(vector<Region> &rs, Mat &region_show)
{
	int maxI = 0;	//the max inertia
	int bg = 0;	//the i of itertia
	for (int i = 0; i < rs.size(); i++)
	{
		rs[i].setJs();
		rs[i].updateRegion(i);
		int ni = rs[i].getJBG();
		rs[i].printInfo(i);
		if (maxI < ni)
		{
			maxI = ni;
			bg = i;
		}
	}
	rs[bg].setBackground();
	vector<Point> pts = rs[bg].getPoints();
	for (int b = 0; b < pts.size(); b++)
	{
		region_show.at<uchar>(pts[b]) = 255;
	}
	return bg;
	//vector<Region>::iterator it = rs.begin() + bg;
	//rs.erase(it);

	//for (int i = 0; i < MAX_REG; i++)
	//{
	//	if (i == bg)
	//	{
	//		rs[i].setBackground();
	//		vector<Point> pts = rs[i].getPoints();
	//		for (int b = 0; b < pts.size(); b++)
	//		{
	//			region_show.at<uchar>(pts[b]) = 255;
	//		}
	//	}
	//	else
	//	{
	//		rs[i].setForeground();
	//	}
	//	cout << i << ":" << rs[i].isBackground() << endl;
	//}
}

void drawEdge(Edge e, Mat &gray, int c)
{
	vector<vector<Point>> fpts = e.getFromRegionPoints();
	int const N = fpts.size();

	for (int i = 0; i < N; i++)
	{
		Point p = fpts[i][0];
		circle(gray, p, 2, c);
	}
	waitKey();
}
void drawEdges(vector<Edge> es, Mat &gray)
{
	for (int i = 0; i < es.size(); i++)
	{
		drawEdge(es[i], gray, i * 50 + 5);
	}
}
float KEdge(Edge e, Mat gray)
{

	float k = 0;
	vector<vector<Point>> fpts = e.getFromRegionPoints();
	vector<vector<Point>> tpts = e.getToRegionPoints();
	int const N = fpts.size();
	float K = 0;
	int fn = e.getFromRegionNumber();
	int tn = e.getToRegionNumber();
	for (int i = 0; i < N; i++)
	{
		for (int ii = 0; ii < EDGE_WIDTH; ii++)
		{
			float F = (float)gray.at<uchar>(fpts[i][ii]);
			float T = (float)gray.at<uchar>(tpts[i][ii]);
			K = K + (float)F / T;
		}
	}
	Mat edg = gray.clone();
	for (int i = 0; i < N; i++)
	{
		Point p = fpts[i][0];
		circle(edg, p, 2, 0);
	}
	imshow("edge used to calculate K", edg);
	//waitKey();
	k = K / EDGE_WIDTH / N;
	std::cout << "K:" << k << endl;
	return k;
}
void  mergeEdges(vector<Edge> &es)
{
	std::cout << "#####start merge edges####" << endl;
	for (int i = 0; i < es.size(); i++)
	{
		std::cout << es[i].getFromRegionNumber() << ":" << es[i].getToRegionNumber() << endl;
	}
	for (int i = 0; i < es.size(); i++)
	{
		int fi = es[i].getFromRegionNumber();
		int ti = es[i].getToRegionNumber();
		vector<vector<Point>> fptsi = es[i].getFromRegionPoints();
		vector<vector<Point>> tptsi = es[i].getToRegionPoints();
		for (int j = 0; j < i; j++)
		{
			int fj = es[j].getFromRegionNumber();
			int tj = es[j].getToRegionNumber();

			//if i-->j , merge i to j and erase i
			if (fj == fi && tj == ti)
			{
				for (int p = 0; p < fptsi.size(); p++)
				{
					es[j].addPoints(fptsi[p], tptsi[p]);
				}
			}
			if (fj == ti && tj == fi)
			{
				for (int p = 0; p < fptsi.size(); p++)
				{
					es[j].addPoints(tptsi[p], fptsi[p]);
				}
			}
			if (fj == ti && tj == fi || fj == fi && tj == ti)
			{
				vector<Edge>::iterator eit = es.begin();
				es.erase(eit + i);
				i--;
				break;
			}
		}
	}
	cout << "#####after merge edges####" << endl;
	for (int i = 0; i < es.size(); i++)
	{
		cout << es[i].getFromRegionNumber() << ":" << es[i].getToRegionNumber() << endl;
	}
	cout << "#####after merge edges####" << endl;
}

void mergeRegions(vector<Region> &rs, vector<Edge> &es, Mat &mygray, Mat &region_map)
{
	for (int i = 0; i < es.size(); i++)
	{//iterate the edges
		Mat mg0 = mygray.clone();
		drawEdges(es, mg0);
		imshow("all edges", mg0);
		//pickup an edge, calculate the k
		Edge ei = es[i];
		float k = KEdge(ei, mygray);
		waitKey();

		//update regions besides to ei(merge)
		int f = ei.getFromRegionNumber();
		int t = ei.getToRegionNumber();
		//cout << i << ":fn-tn-k:" << f << "-" << t << "-" << k << endl;
		int rt;
		for (rt = 0; rt < rs.size(); rt++)
		{
			if (rs[rt].getRegionNum() == t)
				break;
		}
		vector<Point> tpts = rs[rt].getPoints();
		for (int p = 0; p < tpts.size(); p++)
		{
			float old_t = (float)mygray.at<uchar>(tpts[p]);
			float new_t = old_t * k;
			mygray.at<uchar>(tpts[p]) = (uchar)(new_t);
			region_map.at<uchar>(tpts[p]) = (uchar)f;
		}
		rs[rt].setRegionNum(f);
		//remove e
		vector<Edge>::iterator ie = es.begin() + i;
		es.erase(ie);

		//update edges
		for (int ii = 0; ii < es.size(); ii++)
		{
			int erf = es[ii].getFromRegionNumber();
			int ert = es[ii].getToRegionNumber();
			if (erf == t)
			{
				es[ii].setRegions(f, ert);
			}
			if (ert == t)
			{
				es[ii].setRegions(erf, f);
			}
		}
		mergeEdges(es);
		i = -1;
	}
}

int findMin(int a, int b, int c)
{
	int r = a;
	if (r > b)
		r = b;
	if (r > c)
		r = c;
	return r;
}
int findMax(int a, int b, int c)
{
	int r = a;
	if (r < b)
		r = b;
	if (r < c)
		r = c;
	return r;
}
void _main()
{
	Mat img = imread("test2.jpg");
	Mat input_img = img.clone();
	for (int i = 0; i < input_img.rows; i++)
	{
		for (int j = 0; j < input_img.cols; j++)
		{

			int r = input_img.at<Vec3b>(i, j)[0];
			int g = input_img.at<Vec3b>(i, j)[1];
			int b = input_img.at<Vec3b>(i, j)[2];
			if (r < 15 || g < 15 || b < 15)
			{
				r = r * 10;
				g = g * 10;
				b = b * 10;
			}
			float rgb = r + g + b + 1;
			int min = findMax(r, g, b);
			float mean = rgb / 3;
			if (rgb == 1)
				continue;
			int rr = (int)((r + min) / (rgb + 3 * min) * 255);
			int gg = (int)((g + min) / (rgb + 3 * min) * 255);
			int bb = (int)((b + min) / (rgb + 3 * min) * 255);

			if (rr > 255)
				rr = 255;
			if (gg > 255)
				gg = 255;
			if (bb > 255)
				bb = 255;

			input_img.at<Vec3b>(i, j)[0] = r;
			input_img.at<Vec3b>(i, j)[1] = g;
			input_img.at<Vec3b>(i, j)[2] = b;
		}
	}
	imshow("img", img);
	imshow("input", input_img);
	waitKey();
}
void main()
{
	Mat img = imread("c.jpg");
	Mat region_map, region_show, gray;
	cvtColor(img, region_map, CV_RGB2GRAY);
	cvtColor(img, region_show, CV_RGB2GRAY);
	cvtColor(img, gray, CV_RGB2GRAY);

	vector<Region> rs;
	vector<Edge> edges;
	//vector<Mat>  v_hls;
	//cv::split(img, v_hls);
	//imshow("0", v_hls[0]);
	//imshow("1", v_hls[1]);
	//imshow("2", v_hls[2]);
	//waitKey();

	//get regions
	segment(img, rs, edges, region_map, region_show);
	//get regions done

	//remove the background region
	int bg = findBackground(rs, region_show);
	vector<Region>::iterator ir = rs.begin();
	for (; ir != rs.end();)
	{
		if (ir->isBackground())
		{
			rs.erase(ir++);
			break;
		}
		else
			ir++;
	}

	vector<Edge>::iterator ie = edges.begin();
	for (; ie != edges.end();)
	{
		int erf = ie->getFromRegionNumber();
		int ert = ie->getToRegionNumber();
		if (erf == bg || ert == bg)
		{
			ie = edges.erase(ie);
		}
		else
			ie++;
	}
	//remove gackground done

	//using the dominant of RGB channel as the gray scale
	Mat rgb_gray = gray.clone();
	rgb_gray.setTo(255);

	int gc[MAX_REG];
	for (int i = 0; i < rs.size(); i++)
	{
		gc[i] = rs[i].findMaxChannel(img);
		std::cout << "gc[" << i << "]: " << gc[i] << endl;
	}
	for (int i = 0; i < rs.size(); i++)
	{
		vector<Point> rpts = rs[i].getPoints();
		for (int p = 0; p < rpts.size(); p++)
		{
			rgb_gray.at<uchar>(rpts[p]) = img.at<Vec3b>(rpts[p])[gc[i]];
		}

	}


	//result of new method
	Mat mygray = rgb_gray.clone();

	//merge regions
	mergeRegions(rs, edges, mygray, region_map);

	imshow("img", img);
	imshow("gray", gray);
	imshow("region_show", region_show);
	imshow("rgb_gray", rgb_gray);
	imshow("mygray", mygray);
	waitKey(0);
	cout << img.size() << endl;
	system("pause");
}
