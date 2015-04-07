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
	vector<vector<Point>> pp;	//pairs of points on edge
	int rf, rt;	//this is an edge between regins of rf and rt
public:
	void switchPoints(){
		for (int i = 0; i < pp.size(); i++)
		{
			Point p = pp[i][0];
			pp[i][0] = pp[i][1];
			pp[i][1] = p;
		}
	}
	void addPoints(Point pf, Point pt)
	{
		//去重
		for (int i = 0; i < pp.size(); i++)
		{
			Point f = pp[i][0];
			Point t = pp[i][1];
			if (f.x == pf.x && f.y == pf.y)
			{
				if (t.x == pt.x && t.y == pt.y)
					return;
			}
		}
		vector<Point> p;
		p.push_back(pf);
		p.push_back(pt);
		pp.push_back(p);
	}
	void setRegions(int f, int t)
	{
		rf = f;
		rt = t;
	}
	vector<int> getRegionNumbers()
	{
		vector<int> rs;
		rs.push_back(rf);
		rs.push_back(rt);
		return rs;
	}
	vector<vector<Point>> getPoints()
	{
		return pp;
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
	int x = p.x;
	int y = p.y;
	for (int dx = -1; dx < 2; dx++)
	{
		for (int dy = -1; dy < 2; dy++)
		{
			int ex = x + dx;
			int ey = y + dy;
			if (ex < 0)
				ex = 0;
			if (ex > region_map.cols - 1)
				ex = region_map.cols - 1;
			if (ey < 0)
				ey = 0;
			if (ey > region_map.rows - 1)
				ey = region_map.rows - 1;
			int t = (int)region_map.at<uchar>(ey, ex);
			int f = (int)region_map.at<uchar>(y, x);
			if (f != t)
			{
				//相邻两个点属于不同region
				//遍历Edge，把两个点添加到对应region间的edge去
				//如果没有此edge，则建立新的edge

				//遍历
				vector<int> ft;
				int find = 0;
				for (int i = 0; i < edges.size(); i++)
				{
					ft = edges[i].getRegionNumbers();
					//找到edge，添加Points进去
					if (ft[0] == f && ft[1] == t)
					{
						edges[i].addPoints(Point(x, y), Point(ex, ey));
						find = 1;
						break;
					}
					if (ft[0] == t && ft[1] == f)
					{
						edges[i].addPoints(Point(ex, ey), Point(x, y));
						find = 1;
						break;
					}
				}
				//没有找到对应edge，新建edge
				if (!find)
				{
					Edge e;
					e.setRegions(f, t);
					e.addPoints(Point(x, y), Point(ex, ey));
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
	vector<vector<Point>> pts = e.getPoints();
	int const N = pts.size();

	for (int i = 0; i < N; i++)
	{
		Point p = pts[i][0];
		circle(gray, p, 2, c);
	}
	waitKey();
}
void drawEdges(vector<Edge> es, Mat &gray)
{
	for (int i = 0; i < es.size(); i++)
	{
		drawEdge(es[i], gray, i * 50+5);
	}
}
float KEdge(Edge e, Mat gray)
{

	float k = 0;
	vector<vector<Point>> pts = e.getPoints();
	int const N = pts.size();
	float K = 0;
	int fn = e.getRegionNumbers()[0];
	int tn = e.getRegionNumbers()[1];
	for (int i = 0; i < N; i++)
	{
		Point p = pts[i][0];
		int FF = gray.at<uchar>(p.y, p.x);
		float F = (float)gray.at<uchar>(pts[i][0]);
		float T = (float)gray.at<uchar>(pts[i][1]);
		K = K + (float)F / T;
	}
	Mat edg = gray.clone();
	for (int i = 0; i < N; i++)
	{
		Point p = pts[i][0];
		circle(edg, p, 2, 0);
	}
	imshow("edge used to calculate K", edg);
	//waitKey();
	k = K / N;
	cout << "K:" << k << endl;
	return k;
}
void  mergeEdges(vector<Edge> &es)
{
	cout << "#####start merge edges####" << endl;
	for (int i = 0; i < es.size(); i++)
	{
		cout << es[i].getRegionNumbers()[0] << ":" << es[i].getRegionNumbers()[1] << endl;
	}
	for (int i = 0; i < es.size(); i++)
	{
		vector<int> rsi = es[i].getRegionNumbers();
		int fi = rsi[0];
		int ti = rsi[1];
		vector<vector<Point>> ppi = es[i].getPoints();
		for (int j = 0; j < i; j++)
		{
			vector<int> rsj = es[j].getRegionNumbers();
			int fj = rsj[0];
			int tj = rsj[1];
			//if i-->j , merge i to j and erase i
			if (fj == fi && tj == ti)
			{
				for (int p = 0; p < ppi.size(); p++)
				{
					es[j].addPoints(ppi[p][0], ppi[p][1]);
				}
			}
			if (fj == ti && tj == fi)
			{
				for (int p = 0; p < ppi.size(); p++)
				{
					es[j].addPoints(ppi[p][1], ppi[p][0]);
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
		cout << es[i].getRegionNumbers()[0] << ":" << es[i].getRegionNumbers()[1] << endl;
	}
	cout << "#####after merge edges####" << endl;
}
void _mergeEdges(vector<Edge> &es)
{
	cout << "#####start merge edges####" << endl;
	for (int i = 0; i < es.size(); i++)
	{
		cout << es[i].getRegionNumbers()[0] << ":" << es[i].getRegionNumbers()[1] << endl;
	}
	vector<Edge> res;
	int merge = 0;
	for (int i = 0; i < es.size(); i++)
	{
		vector<int> rsi = es[i].getRegionNumbers();
		int fi = rsi[0];
		int ti = rsi[1];
		merge = 0;
		for (int j = 0; j < res.size(); j++)
		{
			vector<int> rsj = res[j].getRegionNumbers();
			int fj = rsj[0];
			int tj = rsj[1];
			if (fj == fi && tj == ti)
			{
				vector<vector<Point>> ppi = es[i].getPoints();
				for (int p = 0; p < ppi.size(); p++)
				{
					res[j].addPoints(ppi[p][0], ppi[p][j]);
				}
			}
			if (fj == ti && tj == fi)
			{
				vector<vector<Point>> ppi = es[i].getPoints();
				for (int p = 0; p < ppi.size(); p++)
				{
					res[j].addPoints(ppi[p][1], ppi[p][0]);
				}
			}
			if (fj == ti && tj == fi || fj == fi && tj == ti)
			{
				cout << i << ":" << "fi:ti->fj:tj" << fi << ":" << ti << "->" << fj << ":" << tj << endl;
				merge = 1;
				vector<Edge>::iterator eit = es.begin();
				es.erase(eit + i);
				break;
			}
		}
		if (0 == merge)
		{
			Edge esi = es[i];
			res.push_back(esi);
			vector<Edge>::iterator eit = es.begin();
			es.erase(eit + i);
		}
	}
	es = res;
	cout << "#####after merge edges####" << endl;
	for (int i = 0; i < es.size(); i++)
	{
		cout << es[i].getRegionNumbers()[0] << ":" << es[i].getRegionNumbers()[1] << endl;
	}
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
		vector<int> ft = ei.getRegionNumbers();
		int f = ft[0];
		int t = ft[1];
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
			vector<int> rn = es[ii].getRegionNumbers();
			if (rn[0] == t)
			{
				es[ii].setRegions(f, rn[1]);
			}
			if (rn[1] == t)
			{
				es[ii].setRegions(rn[0], f);
			}
		}
		mergeEdges(es);
		i = -1;
	}
}

void _mergeRegions(vector<Region> &rs, vector<Edge> &es, Mat &mygray, Mat &region_map)
{
	Mat mg0 = mygray.clone();
	cout << "start merge regions" << endl;
	drawEdges(es, mg0);

	int region_size = rs.size();
	for (int from = 0; from < region_size; from++)
	{//for all regions
		//In fact this loop should be runed for only once, for the reason that 
		//the from region will be extend until all regions is included
		//but in case of some independent regions

		if (rs[from].isBackground())
			//ignore background region
			continue;

		Region &rf = rs[from];
		int fn = rs[from].getRegionNum();
		//mergeEdges(es);
		//find neighbour region by edges
		for (int e = 0; e < es.size(); e++)
		{
			cout << "### " << endl << endl;
			for (int etest = 0; etest < es.size(); etest++)
			{
				vector<int> rtests = es[etest].getRegionNumbers();
				cout << "edge " << etest << ":" << rtests[0] << ":" << rtests[1] << endl;
			}
			vector<int> ft = es[e].getRegionNumbers();
			if ((ft[0] == fn || ft[1] == fn) && (ft[0] != ft[1]))
			{

				//find neighbour
				float k = KEdge(es[e], mygray);
				int tn = ft[1];
				if (ft[1] == fn)
				{
					k = 1 / k;
					tn = ft[0];
				}
				cout << e << ":fn-tn-k:" << fn << "-" << tn << "-" << k << endl;
				//merge tn to fn
				//update the gray img
				//find the region of  tn
				int to = 0;
				for (int i = 0; i < rs.size(); i++)
				{
					if (rs[i].getRegionNum() == tn)
					{
						to = i;
						break;
					}
				}

				vector<Point> tpts = rs[to].getPoints();
				for (int i = 0; i < tpts.size(); i++)
				{
					float old_t = (float)mygray.at<uchar>(tpts[i]);
					float new_t = old_t * k;
					mygray.at<uchar>(tpts[i]) = (uchar)(new_t);
					region_map.at<uchar>(tpts[i]) = (uchar)fn;
				}

				//remove region of tn
				vector<Region>::iterator rit = rs.begin();
				rs.erase(rit + to);

				//remove edge
				vector<Edge>::iterator it = es.begin();
				es.erase(it + e);

				//update edges, change all of tn to fn
				for (int i = 0; i < es.size(); i++)
				{
					vector<int> rn = es[i].getRegionNumbers();
					if (rn[0] == tn)
					{
						es[i].setRegions(fn, rn[1]);
					}
					if (rn[1] == tn)
					{
						es[i].setRegions(rn[0], fn);
					}
				}
				mergeEdges(es);

				//restart from the begining.
				e = 0;
				Mat mg = mygray.clone();
				cout << "merging" << endl;
				drawEdges(es,mg);
				imshow("mygray_test", mygray);
				waitKey();
			}//if ((ft[0] == fn || ft[1] == fn) && (ft[0]!=ft[1]))
		}//for (int e = 0; e < es.size(); e++)
		region_size = rs.size();
	}//for (int from = 0; from < region_size; from++)
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
	Mat img = imread("cn.jpg");
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
		vector<int> ers = ie->getRegionNumbers();
		if (ers[0] == bg || ers[1] == bg)
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
