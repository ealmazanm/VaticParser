#pragma once
#include <stdio.h>
#include <iostream>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <ctype.h> 
#include <fstream>
#include <list>
#include <time.h>
#include <tinystr.h>
#include <tinyxml.h>

using namespace std;
using namespace cv;


Scalar* colorsGT = new Scalar[100];
Scalar* colorsST = new Scalar[100];

const float THRESHOLD_MERGES = 0.005;
const float THRESHOLD_MAHALANOBIS = 30;
const float COV_FACTOR = 2;

struct Frame
{
	int idFrame;
	Rect bbox;
	bool out;
	bool occluded;
	//for DCFs information
	bool used;
	int idTgt;
};

struct FrameEllipse
{
	int idFrame;
	Point mean;
	double covX;
	double covY;
	double covXY;
	bool used;
	int idTgt;
	bool out;
	bool occluded;
};

struct EllipseParam
{
	Point mean;
	double covX;
	double covY;
	double covXY;
};

struct DCF
{
	int idFrame;
	vector<int> trgtIds;
	vector<EllipseParam> gd_Prms;
	EllipseParam avgEllipse;
	/*Point mean;
	double covX;
	double covY;
	double covXY;*/
};

struct Merge
{
	vector<int> tgtIds;
	int initFrame;
	int endFrame;
	int id;
	vector<EllipseParam> gd_Prms;
};

		static void printValuesF(const Mat* m, char* title, ostream& out)
		{
			out << title << endl;
			for (int i = 0; i < m->rows; i++)
			{
				const float* ptr = m->ptr<float>(i);
				for (int j = 0; j < m->cols; j++)
				{
					out << (float)ptr[j] << " ";
				}
				out << endl;
			}
			out << endl;
		}


void fillFrameEllipse(FrameEllipse& fr, char* lineC, int i, int& idTarget)
{
		
	switch (i)
	{
	case 0:
		//id target
		idTarget = atoi(lineC);
		fr.idTgt = idTarget;
		break;
	case 1:
		//MeanX
		fr.mean.x = atoi(lineC);
		break;
	case 2:
		//MeanY
		fr.mean.y = atoi(lineC);
		break;
	case 3:
		//CovX
		fr.covX = atof(lineC);
		break;
	case 4:
		//CovY
		fr.covY = atof(lineC);
		break;
	case 5:
		//CovXY
		fr.covXY = atof(lineC);
		break;
	case 6:
		//id frame
		fr.idFrame = atoi(lineC);
		break;
	case 7:
		//Occlusion
		fr.occluded = atoi(lineC);
		break;
	}
	fr.out = false;
	fr.used = false;

}

void fillFrame(Frame& fr, char* lineC, int i, int& idTarget)
{
	switch (i)
	{
	case 0:
		//id target
		idTarget = atoi(lineC);
		fr.idTgt = idTarget;
		break;
	case 1:
		//Xmin
		fr.bbox.x = atoi(lineC);
		break;
	case 2:
		//Ymin
		fr.bbox.y = atoi(lineC);
		break;
	case 3:
		//width
		fr.bbox.width = atoi(lineC) - fr.bbox.x;
		break;
	case 4:
		//height
		fr.bbox.height = atoi(lineC) - fr.bbox.y;
		break;
	case 5:
		//frameId
		fr.idFrame = atoi(lineC);
		break;
	case 6:
		//out of view
		fr.out = (atoi(lineC) == 1);
		break;
	case 7:
		//occluded
		fr.occluded = (atoi(lineC) == 1);
		break;

	}
	fr.used = false;

}

void initCounters(int* tgtCount, int ttlTgt)
{
	for (int i = 0; i < ttlTgt; i++)
	{
		tgtCount[i] = 0;
	}
}

float mahalanobisDist(FrameEllipse frA, FrameEllipse frB)
{
	float valsA[] = {frA.covX, frA.covXY, frA.covXY, frA.covY};
	Mat covA = Mat(2,2, CV_32F, valsA); 

	float valsB[] = {frB.covX, frB.covXY, frB.covXY, frB.covY};
	Mat covB = Mat(2,2, CV_32F, valsB); 

	float meanValA[] = {frA.mean.x, frA.mean.y};
	Mat meanA = Mat(2,1, CV_32F, meanValA);

	float meanValB[] = {frB.mean.x, frB.mean.y};
	Mat meanB = Mat(2,1, CV_32F, meanValB);

	Mat S = (covA + covB)/2;
	Mat v = (meanA-meanB);
	float dMaha = ((Mat)(v.t() * S.inv() * v)).at<float>(0,0);
	return dMaha;
}

float bhattacharyyaCoeff(FrameEllipse frA, FrameEllipse frB)
{

	float valsA[] = {frA.covX, frA.covXY, frA.covXY, frA.covY};
	Mat covA = Mat(2,2, CV_32F, valsA); 

	float valsB[] = {frB.covX, frB.covXY, frB.covXY, frB.covY};
	Mat covB = Mat(2,2, CV_32F, valsB); 

	float meanValA[] = {frA.mean.x, frA.mean.y};
	Mat meanA = Mat(2,1, CV_32F, meanValA);

	float meanValB[] = {frB.mean.x, frB.mean.y};
	Mat meanB = Mat(2,1, CV_32F, meanValB);

	Mat S = (covA + covB)/2;
	Mat v = (meanA-meanB);
	float dMaha = ((Mat)(v.t() * S.inv() * v)).at<float>(0,0);


	float dBhatt = dMaha/8.0 + log( determinant(S) / sqrt(determinant(covA)* determinant(covB))/2.0 );
	return exp(-dBhatt);

}


void combineEllipses(DCF& dcf)
{
	
	
	dcf.avgEllipse.mean.x = 0; dcf.avgEllipse.mean.y = 0;
	dcf.avgEllipse.covX = 0; dcf.avgEllipse.covY = 0; dcf.avgEllipse.covXY = 0;
//	Mat S = Mat::zeros(2,2, CV_32F);
	int total = dcf.gd_Prms.size();
	for (int i = 0; i < total; i++)
	{
//		float val[] = {dcf.gd_Prms[i].covX, dcf.gd_Prms[i].covXY, dcf.gd_Prms[i].covXY, dcf.gd_Prms[i].covY};
//		Mat cov = Mat(2,2, CV_32F, val);
//		S += cov;
		dcf.avgEllipse.covX += dcf.gd_Prms[i].covX;
		dcf.avgEllipse.covY += dcf.gd_Prms[i].covY;
		dcf.avgEllipse.covXY += dcf.gd_Prms[i].covXY;

		dcf.avgEllipse.mean.x += dcf.gd_Prms[i].mean.x;
		dcf.avgEllipse.mean.y += dcf.gd_Prms[i].mean.y;

	}
	//S /= dcf.gd_Prms.size();
	
	dcf.avgEllipse.mean.x /= total;
	dcf.avgEllipse.mean.y /= total;

	dcf.avgEllipse.covX /= total;
	dcf.avgEllipse.covY /= total;
	dcf.avgEllipse.covXY /= total;

	//TODO: CHECK
	dcf.avgEllipse.covX *= COV_FACTOR;
	dcf.avgEllipse.covY *= COV_FACTOR;

}


void detectDCF_Ellipses(DCF& dcf, int fPosA, vector<FrameEllipse>& framesLst)
{
	FrameEllipse frA = framesLst[fPosA];
	for (int fPosB = 0; fPosB < framesLst.size(); fPosB++)
	{
		if (fPosB == fPosA) continue;
		FrameEllipse frB = framesLst[fPosB];
		//float bCoeff = bhattacharyyaCoeff(frA, frB);
		float mDist = mahalanobisDist(frA, frB);

		if (mDist < THRESHOLD_MAHALANOBIS && !frB.used)
		//if (bCoeff > THRESHOLD_MERGES && !frB.used)
		{
			if (!framesLst[fPosA].used)
			{
				framesLst[fPosA].used = true;
				dcf.trgtIds.push_back(frA.idTgt);	
				EllipseParam ep;
				ep.mean = frA.mean; ep.covX = frA.covX; ep.covY = frA.covY; ep.covXY = frA.covXY;
				dcf.gd_Prms.push_back(ep);
			}
			detectDCF_Ellipses(dcf, fPosB, framesLst);
			if (!framesLst[fPosB].used)
			{
				dcf.trgtIds.push_back(frB.idTgt);
				framesLst[fPosB].used = true;
				EllipseParam ep;
				ep.mean = frB.mean; ep.covX = frB.covX; ep.covY = frB.covY; ep.covXY = frB.covXY;
				dcf.gd_Prms.push_back(ep);

			}
		}
	}
	//Generate the ellipse of the merge meassurement
	if (dcf.gd_Prms.size() > 0)
		combineEllipses(dcf);

}


void detectDCF(DCF& dcf, int fPosA, vector<Frame>& framesLst)
{
	Frame frA = framesLst[fPosA];
	for (int fPosB = 0; fPosB < framesLst.size(); fPosB++)
	{
		if (fPosB == fPosA) continue;
		Frame frB = framesLst[fPosB];
		Rect overlap = frA.bbox & frB.bbox;
		if (overlap.width > 0 && overlap.height > 0 && !frB.used)
		{
			if (!framesLst[fPosA].used)
			{
				framesLst[fPosA].used = true;
				dcf.trgtIds.push_back(frA.idTgt);				
			}
			detectDCF(dcf, fPosB, framesLst);
			if (!framesLst[fPosB].used)
			{
				dcf.trgtIds.push_back(frB.idTgt);
				framesLst[fPosB].used = true;
			}
			
			

		}
	}

}

/* 
Search through all DCFs for fr
*/
bool isDCF(FrameEllipse fr, vector<DCF> dcfsCurrent)
{

	bool found = false;
	int cont = 0;
	int total = dcfsCurrent.size();
	while (cont < total && !found)
	{
		DCF dcf = dcfsCurrent[cont++];
		found = find(dcf.trgtIds.begin(), dcf.trgtIds.end(), fr.idTgt)!=dcf.trgtIds.end();
	}
	return found;
}

bool isDCF(vector<DCF> dcfs, int frId)
{
	int total = dcfs.size();
	bool found = false;
	int cont = 0;
	while (!found && cont < total)
	{
		found = dcfs[cont++].idFrame == frId;
	}
	return found;
}

bool isDCF(Frame fr, vector<DCF> dcfsCurrent)
{

	bool found = false;
	int cont = 0;
	int total = dcfsCurrent.size();
	while (cont < total && !found)
	{
		DCF dcf = dcfsCurrent[cont++];
		found = find(dcf.trgtIds.begin(), dcf.trgtIds.end(), fr.idTgt)!=dcf.trgtIds.end();
	}
	return found;
}


int getEllipseParams(const Mat* frame, Rect bbox, Point& center, double& covX, double& covY, double& covXY)
{
	double x, y, xx, yy, xy, w;
	x = y = xx = yy = xy = w = 0;
	int endY = bbox.y + bbox.height;
	int endX = bbox.x + bbox.width;
	for (int i = bbox.y; i < endY; i++)
	{
		const uchar* ptr = frame->ptr<uchar>(i);
		for (int j = bbox.x; j < endX; j++)
		{
			if (ptr[3*j] < 50 && ptr[3*j+1] < 50 && ptr[3*j+2] < 50)
			{
				x += j;
				y += i;
				w++;
				xx += j*j;
				yy += i*i; 
				xy += i*j;
			}
		}
	}

	double meanX = x/w;
	double meanY = y/w;
	center.x = meanX; 
	center.y = meanY;

	covX = (xx/w) - (meanX*meanX);
	covY = (yy/w) - (meanY*meanY);
	covXY = (xy/w) - (meanX*meanY);

	return 1;
}




void getEllipseParams2(int& bigAxis, int& smallAxis, double& angle, double& covX, double& covY, double& covXY)
{
	float covValues[] = {covX, covXY, covXY, covY};
	Mat cov = Mat(2,2, CV_32F, covValues);

	//printValuesF(&cov, "covariance", cout);

	SVD svd(cov);
	
	bigAxis = sqrtf(svd.w.at<float>(0))*2.5;
	smallAxis = sqrtf(svd.w.at<float>(1))*2.5;

	//identify the quadrant of the main eigenvector
	bool upperQuadrant = (svd.u.at<float>(1,0) > 0);
	Mat bigEigenVct = svd.u(Rect(0,0, 1,2));
	float vals[] = {1, 0};
	Mat mainAxis = Mat(2,1, CV_32F, vals);
	float dotPrd = bigEigenVct.dot(mainAxis);
	angle = acosf(dotPrd)*180/CV_PI;
	
	if (!upperQuadrant)
					angle = -angle;

}


void getEllipseParams(const Mat* frame, Rect bbox, Point& center, int& bigAxis, int& smallAxis, double& angle, double& covX, double& covY, double& covXY)
{
	double x, y, xx, yy, xy, w;
	x = y = xx = yy = xy = w = 0;
	int endY = bbox.y + bbox.height;
	int endX = bbox.x + bbox.width;
	for (int i = bbox.y; i < endY; i++)
	{
		const uchar* ptr = frame->ptr<uchar>(i);
		for (int j = bbox.x; j < endX; j++)
		{
			if (ptr[3*j] < 30 && ptr[3*j+1] < 30 && ptr[3*j+2] < 30)
			{
				x += j;
				y += i;
				w++;
				xx += j*j;
				yy += i*i; 
				xy += i*j;
			}
		}
	}

	double meanX = x/w;
	double meanY = y/w;
	center.x = meanX; 
	center.y = meanY;

	covX = (xx/w) - (meanX*meanX);
	covY = (yy/w) - (meanY*meanY);
	covXY = (xy/w) - (meanX*meanY);

	float covValues[] = {covX, covXY, covXY, covY};
	Mat cov = Mat(2,2, CV_32F, covValues);

	//printValuesF(&cov, "covariance", cout);

	SVD svd(cov);
	
	bigAxis = sqrtf(svd.w.at<float>(0))*2.5;
	smallAxis = sqrtf(svd.w.at<float>(1))*2.5;

	//identify the quadrant of the main eigenvector
	bool upperQuadrant = (svd.u.at<float>(1,0) > 0);
	Mat bigEigenVct = svd.u(Rect(0,0, 1,2));
	float vals[] = {1, 0};
	Mat mainAxis = Mat(2,1, CV_32F, vals);
	float dotPrd = bigEigenVct.dot(mainAxis);
	angle = acosf(dotPrd)*180/CV_PI;
	
	if (!upperQuadrant)
					angle = -angle;



}

const char* VATIC_GT_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\gt.txt";
const char* ELLIPSE_GT_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\gt_ellipses.txt";
const char* DCFs_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\dcfs.xml";
const char* STMERGES_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\mergeMeasurements_system.xml";
const char* DCFs_FILE_TXT = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\dcfs.txt";
const char* MERGE_GT_FILE_XML = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\mergeMeasurements_gt.xml";

const char* MOA_ORIGINAL_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\MoA_Detection_final.mpg";
const char* MOA_GT_LABELLED_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputDCF.avi";
const char* MOA_ST_LABELLED_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputST.avi";
const char* MOA_GT_MERGES_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputMerges.avi";
const char* MOA_GT_ST_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputgtst.avi";

const int THRESHOLD_MERGE = 5;


void readEllipseFile(ifstream& inEllipseGt, vector<vector<FrameEllipse>>&gtEllipseData, bool gt)
{
	string line;
	int idTgtPrev = -1;
	int c = 0;

	Scalar* colors;
	if (gt)
		colors = colorsGT;
	else
		colors = colorsST;

	while (getline(inEllipseGt, line))
	{
		char *lineC = new char[line.length() + 1];
		strcpy(lineC, line.c_str());
		lineC = strtok(lineC, " ");
		int i = 0;

		int idTarget;
		FrameEllipse fr;
		
		while (lineC)
		{
			
			//printf ("Token %d: %s\n", i,lineC);
			fillFrameEllipse(fr, lineC, i++, idTarget);
			lineC = strtok(NULL, " ");
			if (colors[idTarget].val[0] = -1)
			{
				int red = rand() % 255 + 1;
				int green = rand() % 255 + 1;
				int blue = rand() % 255 + 1;
				Scalar c = Scalar(red,green, blue);
				colors[idTarget] = c;
			}	
		}


		if (idTarget >= gtEllipseData.size())
			gtEllipseData.resize(idTarget+1);

		gtEllipseData[idTarget].push_back(fr);
		delete [] lineC;
	}
}

/*
Read the ground truth file generated by VATIC of bounding boxes of all targets in every frame of the sequence.
out: List of targets, and each target contains a list of all the frames where is present in the sequence. Each frame contain the bounding box information.
*/
void readVaticGtFile(ifstream& inGt, vector<vector<Frame>>& gtData)
{
	string line;
	int idTgtPrev = -1;
	while (getline(inGt, line))
	{
		char *lineC = new char[line.length() + 1];
		strcpy(lineC, line.c_str());

		lineC = strtok(lineC, " ");
		int i = 0;

		int idTarget;
		Frame fr;
		while (lineC)
		{
			//printf ("Token %d: %s\n", i,lineC);
			fillFrame(fr, lineC, i++, idTarget);
			lineC = strtok(NULL, " ");
		}
		if (idTgtPrev != idTarget)
		{
			vector<Frame> frames;
			gtData.push_back(frames);
		}

		gtData[idTarget].push_back(fr);

		idTgtPrev = idTarget;
		delete [] lineC;
	}
}


bool isOnCurrenList(list<Merge>& currentList, Merge* mOld)
{
	Merge* m = NULL;

	bool out = false;
	list<Merge>::iterator iter = currentList.begin();
	while ( iter != currentList.end() && !out)
	{
		Merge &mC( *iter );
		if ((mC.tgtIds == mOld->tgtIds))
		{
			mC.initFrame = mOld->initFrame;
			mC.id = mOld->id;
			if (mC.gd_Prms.size() != 1)
				cout << "Error" << endl;
			EllipseParam ep = mC.gd_Prms[0];
			mOld->gd_Prms.push_back(ep);
			mC.gd_Prms = mOld->gd_Prms;
			out =  true;
		}

		iter++;
	}
	return out;
}

int getId(list<Merge>& mergeList)
{
	int idMax = -1;
	list<Merge>::iterator iter = mergeList.begin();
	while (iter != mergeList.end())
	{
		Merge m = *iter;
		if (m.id > idMax)
			idMax = m.id;

		iter++;
	}
	return idMax + 1;
}

//M is defined before than s in time
void addMissingEllipses(Merge& m, Merge& s, vector<vector<FrameEllipse>>& gtData)
{
	Point mean;
	double covX, covY, covXY;
	for (int i = m.endFrame + 1; i < s.initFrame; i++)
	{
		mean.x = 0; mean.y = 0;
		covX = covY = covXY = 0;

		int ttlTgts = m.tgtIds.size();
		for (int j = 0; j < ttlTgts; j++)
		{
			int tgtId = m.tgtIds[j];
			bool found = false;
			int gtIter = -1;
			int ttlGtData = gtData[tgtId].size();
			while (!found && gtIter <= ttlGtData)
			{
				gtIter++;
				FrameEllipse fp = gtData[tgtId][gtIter];
				found = (i == fp.idFrame);
				if (found)
				{
					mean.x += fp.mean.x;
					mean.y += fp.mean.y;
					covX += fp.covX;
					covY += fp.covY;
					covXY += fp.covXY;
				}
			}
		}
		EllipseParam ep;
		ep.mean.x =  mean.x / ttlTgts; 
		ep.mean.y = mean.y / ttlTgts;
		ep.covX = covX / ttlTgts;
		ep.covY = covY / ttlTgts;
		ep.covXY = covXY / ttlTgts;

		ep.covX *= COV_FACTOR;
		ep.covY *= COV_FACTOR;
		m.gd_Prms.push_back(ep);
	}

	//concatenate the two vectors m and s
	m.gd_Prms.insert(m.gd_Prms.end(), s.gd_Prms.begin(), s.gd_Prms.end());

}

void postProcess(list<Merge>& mergeList, list<Merge>& mergeListPost, vector<vector<FrameEllipse>>& gtData)
{

	list<Merge>::iterator iter1 = mergeList.begin();
	int mergeId = 0;
	while (iter1 != mergeList.end())
	{
		Merge m = *iter1;
		m.id = mergeId++;
		list<Merge>::iterator iter2 = iter1;
		++iter2;
		while (iter2 != mergeList.end())
		{
			Merge s = *iter2;
			if (m.tgtIds == s.tgtIds)
			{
				if (s.initFrame >= m.endFrame)
				{
					if ((s.initFrame - m.endFrame) < THRESHOLD_MERGE)
					{
						addMissingEllipses(m, s, gtData);
						m.endFrame = s.endFrame;
						
						iter2 = mergeList.erase(iter2);
					}
					else 
						iter2++;
				}
				else
					if ((m.initFrame - s.endFrame) < THRESHOLD_MERGE)
					{
						addMissingEllipses(s, m, gtData);
						s.endFrame = m.endFrame;
						m = s;
						iter2 = mergeList.erase(iter2);
					}				
					else
						iter2++;
			}
			else
				iter2++;
		}
		mergeListPost.push_back(m);
		iter1++;
	}
}

//Assumes DCFs are ordered according to the frame ides
void detectMerges(vector<DCF>& DCFs, list<Merge>& mergeList)
{

	list<Merge> oldList;
	int idDCF = 0;
	
	for (int frId = 0; frId < 1000; frId++)
	{
		list<Merge> currentList;
		int mergeIds = mergeList.size() + oldList.size();

		while (idDCF < DCFs.size() && DCFs[idDCF].idFrame == frId)
		{
			DCF dcf = DCFs[idDCF++];
			Merge m;
			m.id = mergeIds++;
			m.initFrame = frId;
			m.tgtIds = dcf.trgtIds;
			m.gd_Prms.push_back(dcf.avgEllipse);
			currentList.push_front(m);
		}
		list<Merge>::iterator iterOld = oldList.begin();
		while (iterOld != oldList.end())
		{
			Merge mOld = *iterOld;
			if (!isOnCurrenList(currentList, &mOld))
			{
				mOld.endFrame = frId-1;
				mOld.id = getId(mergeList);
				mergeList.push_back(mOld);
				iterOld = oldList.erase(iterOld);
				
			}
			else
			{
				iterOld++;
			}
		}
		
		oldList = currentList;
	}

	//Sanity check
	list<Merge>::iterator iter = mergeList.begin();
	while (iter != mergeList.end())
	{
		Merge m = *iter;
		int ttlEllipses = m.gd_Prms.size();
		int ttlFrames = (m.endFrame - m.initFrame) + 1;
		if (ttlEllipses != ttlFrames)
			cout << "Error" << endl;

		iter++;
	}

}

void dcfDetectionEllipse(vector<DCF>& DCFs, vector<vector<FrameEllipse>>& gtData)
{
	int ttlTgt = gtData.size();

	//go through all frames
	int* tgtCount = new int[ttlTgt]; //keep track of the last frame analyze for each target
	initCounters(tgtCount, ttlTgt);

	////Add xml header <DCFs>
	//TiXmlDocument doc; 
	//TiXmlElement * root = new TiXmlElement( "DCFs" );  
	//doc.LinkEndChild( root );  
	for (int i = 0; i < 1000; i++)
	{
		////Add to xml File
		//TiXmlElement * dcfXml = new TiXmlElement( "DCF" );  
		//dcfXml->SetAttribute("frameId", i);
		//root->LinkEndChild( dcfXml );  

		//cout << "Frame: " << i << endl
		vector<FrameEllipse> tgtFrames; //stores the frames of the targets that are present in the current frame
		for (int j = 0; j < ttlTgt; j++)
		{
			int idCount = tgtCount[j];
			if (idCount >= gtData[j].size())
				continue;

			FrameEllipse f = gtData[j][idCount];
			if (f.idFrame == i) //if the target is present in the current frame
			{
				if (!f.out)
					tgtFrames.push_back(f); //position of targets that are present in the current frame
					
				tgtCount[j]++; //Position of the list of frames of the target.
			}
		}
	
		//go through all targets present in the current frame
		int contXml = 0;
		for (int j = 0; j < tgtFrames.size(); j++)
		{
			if (!tgtFrames[j].used)
			{
				DCF dcf;

				detectDCF_Ellipses(dcf, j, tgtFrames);
				if (dcf.trgtIds.size() > 0)
				{
					dcf.idFrame = i;
					DCFs.push_back(dcf);

					////Add to xml File
					//TiXmlElement * occXml = new TiXmlElement( "Occlusion" );  
					//occXml->SetAttribute("id", contXml++);					
					//for (int k = 0; k < dcf.trgtIds.size(); k++)
					//{
					//	//TiXmlElement * tgtXml = new TiXmlElement( "Target" ); 
					//	//tgtXml->SetAttribute("Id", dcf.trgtIds[k]);
					//	//occXml->LinkEndChild(tgtXml);
					//}
					//dcfXml->LinkEndChild(occXml);
				}
			}
		}
	}
	//doc.SaveFile( "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\dcfs.xml" );  
}


void dcfDetection(vector<DCF>& DCFs, vector<vector<Frame>>& gtData)
{
	int ttlTgt = gtData.size();

	//go through all frames
	int* tgtCount = new int[ttlTgt]; //keep track of the last frame analyze for each target
	initCounters(tgtCount, ttlTgt);

	////Add xml header <DCFs>
	//TiXmlDocument doc; 
	//TiXmlElement * root = new TiXmlElement( "DCFs" );  
	//doc.LinkEndChild( root );  
	for (int i = 0; i < 1000; i++)
	{
		////Add to xml File
		//TiXmlElement * dcfXml = new TiXmlElement( "DCF" );  
		//dcfXml->SetAttribute("frameId", i);
		//root->LinkEndChild( dcfXml );  

		//cout << "Frame: " << i << endl
		vector<Frame> tgtFrames; //stores the frames of the targets that are present in the current frame
		for (int j = 0; j < ttlTgt; j++)
		{
			int idCount = tgtCount[j];
			if (idCount >= gtData[j].size())
				continue;

			Frame f = gtData[j][idCount];
			if (f.idFrame == i) //if the target is present in the current frame
			{
				if (!f.out)
					tgtFrames.push_back(f); //position of targets that are present in the current frame
					
				tgtCount[j]++; //Position of the list of frames of the target.
			}
		}
	
		//go through all targets present in the current frame
		int contXml = 0;
		for (int j = 0; j < tgtFrames.size(); j++)
		{
			if (!tgtFrames[j].used)
			{
				DCF dcf;

				detectDCF(dcf, j, tgtFrames);
				if (dcf.trgtIds.size() > 0)
				{
					dcf.idFrame = i;
					DCFs.push_back(dcf);

					////Add to xml File
					//TiXmlElement * occXml = new TiXmlElement( "Occlusion" );  
					//occXml->SetAttribute("id", contXml++);					
					//for (int k = 0; k < dcf.trgtIds.size(); k++)
					//{
					//	//TiXmlElement * tgtXml = new TiXmlElement( "Target" ); 
					//	//tgtXml->SetAttribute("Id", dcf.trgtIds[k]);
					//	//occXml->LinkEndChild(tgtXml);
					//}
					//dcfXml->LinkEndChild(occXml);
				}
			}
		}
	}
	//doc.SaveFile( "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\dcfs.xml" );  
}

void readXMLSMerges(list<Merge>& mergeListST, const char* path)
{
	TiXmlDocument doc (path);
	if (doc.LoadFile())
	{
		TiXmlElement *pRoot, *pMerge, *pTarget, *pFrame;
		pRoot = doc.FirstChildElement("Merges");
		if (pRoot)
		{
			pMerge = pRoot->FirstChildElement("Merge");
			while (pMerge)
			{
				Merge m;
				m.id = atoi(pMerge->Attribute("Id"));
				m.initFrame = atoi(pMerge->Attribute("InitFrame"));
				m.endFrame = atoi(pMerge->Attribute("EndFrame"));
				
				pTarget = pMerge->FirstChildElement("Target");
				while (pTarget)
				{
					m.tgtIds.push_back(atoi(pTarget->Attribute("Id")));
					pTarget = pTarget->NextSiblingElement("Target");
				}

				pFrame = pMerge->FirstChildElement("Frames")->FirstChildElement("Frame");
				while (pFrame)
				{
					EllipseParam ep;
					ep.mean.x = atoi(pFrame->Attribute("meanX"));
					ep.mean.y = atoi(pFrame->Attribute("meanY"));
					ep.covX = atof(pFrame->Attribute("covX"));
					ep.covY = atof(pFrame->Attribute("covY"));
					ep.covXY = atof(pFrame->Attribute("covXY"));
					m.gd_Prms.push_back(ep);
					pFrame = pFrame->NextSiblingElement("Frame");
				}
				mergeListST.push_back(m);

				pMerge = pMerge->NextSiblingElement("Merge");
			}
		}
	}
}

void readDCFsFile(vector<DCF>& DCFs_II)
{
	TiXmlDocument doc( DCFs_FILE );
	if (doc.LoadFile())
	{
		TiXmlElement *pRoot, *pDCF, *pOccl, *pTgt;
		pRoot = doc.FirstChildElement("DCFs");
		if (pRoot)
		{
			pDCF = pRoot->FirstChildElement("DCF");
			while (pDCF)
			{
				int idFrame = atoi(pDCF->Attribute("frameId"));
				pOccl = pDCF->FirstChildElement("Occlusion");
				while (pOccl)
				{
					int idOccl = atoi(pOccl->Attribute("id"));
					DCF dcf;
					dcf.idFrame = idFrame;

					pTgt = pOccl->FirstChildElement("Target");
					while(pTgt)
					{
						int tgtId = atoi(pTgt->Attribute("Id"));
						dcf.trgtIds.push_back(tgtId);
						pTgt = pTgt->NextSiblingElement("Target");
					}
					DCFs_II.push_back(dcf);
					pOccl = pOccl->NextSiblingElement("Occlusion");
				}
				pDCF = pDCF->NextSiblingElement("DCF");
			}
		}
	}
	else
	{
		cerr << "Could not load XML file " << DCFs_FILE << endl;
	}

}

void writeTXTDCFs(const vector<DCF>& DCFs)
{
	ofstream inDCFs (DCFs_FILE_TXT);

	int idDCF = 0;
	for (int frId = 0; frId < 1000; frId++)
	{
		int nDCF = 0;
		while (idDCF < DCFs.size() && DCFs[idDCF].idFrame == frId)
		{
			nDCF++;
			idDCF++;
		}
		inDCFs << nDCF << endl;
	}
	inDCFs.close();

}


void writeXMLMerges(list<Merge>& mergeList)
{
	//Add xml header <DCFs>
	TiXmlDocument doc; 
	TiXmlElement * root = new TiXmlElement( "Merges" );  
	doc.LinkEndChild( root );

	list<Merge>::iterator iter = mergeList.begin();
	while (iter != mergeList.end())
	{
		Merge m = *iter;
		TiXmlElement * mergeXml = new TiXmlElement( "Merge" ); 
		root->LinkEndChild( mergeXml );
		mergeXml->SetAttribute("Id", m.id);
		mergeXml->SetAttribute("InitFrame", m.initFrame);
		mergeXml->SetAttribute("EndFrame", m.endFrame);
		int ttlTgts = m.tgtIds.size();
		for (int i = 0; i < ttlTgts; i++)
		{
			TiXmlElement * tgtXML = new TiXmlElement( "Target" ); 
			int tgtId = m.tgtIds[i];
			tgtXML->SetAttribute("Id", tgtId);
			mergeXml->LinkEndChild(tgtXML);
		}

		TiXmlElement * framesXML = new TiXmlElement( "Frames" ); 
		for (int i = m.initFrame; i <= m.endFrame; i++)
		{
			TiXmlElement * frameXML = new TiXmlElement( "Frame" ); 
			EllipseParam ep = m.gd_Prms[i-m.initFrame];
			frameXML->SetAttribute("Number", i);
			frameXML->SetAttribute("meanX", ep.mean.x);
			frameXML->SetAttribute("meanY", ep.mean.y);
			frameXML->SetDoubleAttribute("covX", ep.covX);
			frameXML->SetDoubleAttribute("covY", ep.covY);
			frameXML->SetDoubleAttribute("covXY", ep.covXY);
			framesXML->LinkEndChild(frameXML);
		}
		mergeXml->LinkEndChild(framesXML);
		iter++;
		
	}
	doc.SaveFile( MERGE_GT_FILE_XML );  
}

void writeXMLDCFs(const vector<DCF>& DCFs)
{
	//Add xml header <DCFs>
	TiXmlDocument doc; 
	TiXmlElement * root = new TiXmlElement( "DCFs" );  
	doc.LinkEndChild( root );  
	int idDCF = 0;
	for (int frId = 0; frId < 1000; frId++)
	{
		//Add to xml File
		TiXmlElement * dcfXml = new TiXmlElement( "DCF" );  
		dcfXml->SetAttribute("frameId", frId);
		root->LinkEndChild( dcfXml );  
	
		int cont = 0;
		while (idDCF < DCFs.size() && DCFs[idDCF].idFrame == frId)
		{
			DCF dcf = DCFs[idDCF];
			//Add to xml File
			TiXmlElement * occXml = new TiXmlElement( "Occlusion" );  
			occXml->SetAttribute("id", cont++);			
			for (int i = 0; i < dcf.trgtIds.size(); i++)
			{
				TiXmlElement * tgtXml = new TiXmlElement( "Target" ); 
				tgtXml->SetAttribute("Id", dcf.trgtIds[i]);
				occXml->LinkEndChild(tgtXml);
			}
			dcfXml->LinkEndChild(occXml);
			idDCF++;
		}
	}
	doc.SaveFile( DCFs_FILE );  
}

void writeXMLEllipseGT(vector<vector<Frame>>& gtData, ofstream& outGt)
{
	int ttlTgt = gtData.size();
	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		int idDCF = 0;
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;
			
			for (int i = 0; i < ttlTgt; i++)
			{
				vector<Frame> tgt = gtData[i];

				bool found = false;
				int cont = 0;
				Frame fr_;
				int ttlFrames = tgt.size();
				while (!found && cont < ttlFrames)
				{
					fr_ = tgt[cont++];
					found = (fr_.idFrame >= frId) && (!fr_.out);
				}
				if (found && fr_.idFrame == frId)
				{
					Point center;
					double covX, covY, covXY;
					covX = covY = covXY = 0;
					int ok = 0;
					ok = getEllipseParams(&frame, fr_.bbox, center, covX, covY, covXY);
					//save in file
					if (ok == 1)
						outGt << i << " " << center.x << " " << center.y << " " << covX << " " << covY << " " << covXY << " " << frId << " " << fr_.occluded << endl;				
					else
						cout << "ERROR" << endl;
				}
			}
			frId++;
		}
	}
	cap.release();
}



bool drawEllipses(list<Merge>& mergeList, Mat& frame, int frId, Scalar color)
{
	bool drawn = false;
	list<Merge>::iterator iter = mergeList.begin();
	while (iter != mergeList.end())
	{
		Merge m = *iter;
				
		if (m.initFrame <= frId && m.endFrame >= frId)
		{

			int pos = frId - m.initFrame;
			EllipseParam ep = m.gd_Prms[pos];
									
			int bigAxis = 0; 
			int smallAxis = 0;
			double angle = 0;
			getEllipseParams2(bigAxis, smallAxis, angle, ep.covX, ep.covY, ep.covXY);
			ellipse(frame, ep.mean, Size(bigAxis, smallAxis), angle, 0, 360, color, 3);
			drawn = true;
		}
		//else if (frId  == (m.initFrame-3) || frId == (m.endFrame+3))
		//{
		//	int pos = 0;
		//	if (frId == (m.endFrame+3))
		//		pos = m.endFrame - m.initFrame;
		//	
		//	EllipseParam ep = m.gd_Prms[pos];
		//							
		//	int bigAxis = 0; 
		//	int smallAxis = 0;
		//	double angle = 0;
		//	getEllipseParams2(bigAxis, smallAxis, angle, ep.covX, ep.covY, ep.covXY);
		//	ellipse(frame, ep.mean, Size(bigAxis, smallAxis), angle, 0, 360, Scalar(0,255,0), 3);
		//	drawn = true;
		//}




		iter++;
	}
	return drawn;
}


void drawTrack(Mat& frame, vector<vector<FrameEllipse>>& track, bool elps, int frId)
{
	Scalar* colors;
	if (!elps)
		colors = colorsGT;
	else
		colors = colorsST;

	//Draw bboxex
	int ttlTgt = track.size();
	for (int i = 0; i < ttlTgt; i++)
	{
		vector<FrameEllipse> tgt = track[i];

		bool found = false;
		int cont = 0;
		FrameEllipse fr_;
		int ttlFrames = tgt.size();
		while (!found && cont < ttlFrames)
		{
			fr_ = tgt[cont++];
			found = (fr_.idFrame >= frId) && (!fr_.occluded);
		}
		if (found && fr_.idFrame == frId)
		{
			char id_str[10];
			itoa(fr_.idTgt, id_str, 10);

			Scalar c = colors[fr_.idTgt];
			double angle;
			int bigAxis = 0; 
			int smallAxis = 0;
			angle = 0;
			double covX, covY, covXY;
			getEllipseParams2(bigAxis, smallAxis, angle, fr_.covX, fr_.covY, fr_.covXY);
			if (elps)
			{
				ellipse(frame, fr_.mean, Size(bigAxis, smallAxis), angle, 0, 360, c, 2);
				putText(frame, id_str, Point(fr_.mean.x + bigAxis, fr_.mean.y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, c);
			}
			else
			{
				circle(frame, Point(fr_.mean.x, fr_.mean.y), 2, c);
				putText(frame, id_str, Point(fr_.mean.x - 40, fr_.mean.y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, c);
			}

			
		}
	}
}


void drawST_GTtracks(vector<vector<FrameEllipse>>& gt, vector<vector<FrameEllipse>>& st)
{

	VideoWriter w(MOA_GT_ST_FILE,CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);

	int waitGt = 1;
	int waitSt = 1;
	int wait = 1;
	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;

			drawTrack(frame, gt, false, frId);
			drawTrack(frame, st, true, frId);

			w << frame;
			cout << "Frame id: " << frId << endl;
			
			imshow("MoA", frame);
			int c = waitKey(wait);
			if (c == 13)
				wait = 0;

			frId++;
		}
	}
	cap.release();
}

void drawGtmerges_stracks(list<Merge>& mergeListGt, vector<vector<FrameEllipse>> stEllipseData)
{
	int waitGt = 1;
	int waitSt = 1;
	int wait = 1;
	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;


			waitGt = !drawEllipses(mergeListGt, frame, frId, Scalar(255, 0, 0));
			drawTrack(frame, stEllipseData, true, frId);

			cout << "Frame id: " << frId << endl;
			
			imshow("MoA", frame);
			if (waitGt == 0)
				wait = waitGt;

			if (frId == 380)
				wait = 0;
			int c = waitKey(wait);
			if (c == 13)
				wait = !wait;

			frId++;
		}
	}
	cap.release();

}

void drawS_GTmerges(list<Merge>& mergeListGt, list<Merge>& mergeListSt)
{
	int waitGt = 1;
	int waitSt = 1;
	int wait = 1;
	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;


			waitGt =  !drawEllipses(mergeListGt, frame, frId, Scalar(255, 0, 0));
			waitSt = !drawEllipses(mergeListSt, frame, frId, Scalar(0, 0, 255));
			wait = min(waitGt, waitSt);

			cout << "Frame id: " << frId << endl;
			if (frId == 430)
				wait = 1;

			imshow("MoA", frame);
			int c = waitKey(wait);
			if (c == 13)
				wait = 1;

			frId++;
		}
	}
	cap.release();
}

void drawMerges(list<Merge>& mergeListPost)
{
	VideoWriter w(MOA_GT_MERGES_FILE,CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);
	//VideoWriter w(MOA_GT_MERGES_FILE,CV_FOURCC('M','J','P','G'), 10.0, Size(1148,480), true);

	int wait = 1;

	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;


			vector<Merge> mergesCurrent;
			list<Merge>::iterator iter = mergeListPost.begin();
			while (iter != mergeListPost.end())
			{
				Merge m = *iter;
				
				if (m.initFrame <= frId && m.endFrame >= frId)
				{
					int pos = frId - m.initFrame;
					EllipseParam ep = m.gd_Prms[pos];
									
					int bigAxis = 0; 
					int smallAxis = 0;
					double angle = 0;
					getEllipseParams2(bigAxis, smallAxis, angle, ep.covX, ep.covY, ep.covXY);
					ellipse(frame, ep.mean, Size(bigAxis, smallAxis), angle, 0, 360, Scalar(0, 0 , 255), 3);
					wait = 0;
				}
				iter++;
			}			
			w << frame;
			cout << "Frame id: " << frId << endl;
			if (frId == 430)
				wait = 1;

			imshow("MoA", frame);
			int c = waitKey(wait);
			if (c == 13)
				wait = 1;

			frId++;
		}

	}
	cap.release();
}



void drawEllipses(vector<vector<FrameEllipse>>& gtData, vector<DCF>& DCFs)
{
	//VideoWriter w(MOA_GT_LABELLED_FILE,CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);
	VideoWriter w(MOA_GT_LABELLED_FILE,CV_FOURCC('M','J','P','G'), 10.0, Size(1148,480), true);
	//VideoWriter w(MOA_ST_LABELLED_FILE,CV_FOURCC('M','J','P','G'), 10.0, Size(1148,480), true);
	
	int wait = 1;
	int ttlTgt = gtData.size();
	VideoCapture cap(MOA_ORIGINAL_FILE);
	int frId = 0;
	if (cap.isOpened())
	{
		int idDCF = 0;
		while(frId < 1000)
		{
			Mat frame;
			cap >> frame;

			vector<DCF> dcfsCurrent;
			while (idDCF < DCFs.size() && DCFs[idDCF].idFrame == frId)
			{
				DCF dcf = DCFs[idDCF];

				int bigAxis = 0; 
				int smallAxis = 0;
				double angle = 0;
				getEllipseParams2(bigAxis, smallAxis, angle, dcf.avgEllipse.covX, dcf.avgEllipse.covY, dcf.avgEllipse.covXY);
				//ellipse(frame, dcf.avgEllipse.mean, Size(bigAxis, smallAxis), angle, 0, 360, Scalar::all(0), 3);

				cout << "DCF: " ;
				for (int i = 0; i < dcf.trgtIds.size(); i++)
				{
					cout << dcf.trgtIds[i] << " ";
				}
				cout << endl;
				dcfsCurrent.push_back(dcf);
				//wait = 0;
				idDCF++;
			}
	

			//Draw bboxex
			for (int i = 0; i < ttlTgt; i++)
			{
				vector<FrameEllipse> tgt = gtData[i];

				bool found = false;
				int cont = 0;
				FrameEllipse fr_;
				int ttlFrames = tgt.size();
				while (!found && cont < ttlFrames)
				{
					fr_ = tgt[cont++];
					found = (fr_.idFrame >= frId) && (!fr_.occluded);
				}
				if (found && fr_.idFrame == frId)
				{
					Scalar c = colorsGT[fr_.idTgt];
					
					//if (isDCF(fr_, dcfsCurrent))
					//	c = Scalar(0,255,0);
					//if (isDCF(DCFs, frId))
					//	wait = 0;

					double angle;
					int bigAxis = 0; 
					int smallAxis = 0;
					angle = 0;
					double covX, covY, covXY;
					getEllipseParams2(bigAxis, smallAxis, angle, fr_.covX, fr_.covY, fr_.covXY);
					
					ellipse(frame, fr_.mean, Size(bigAxis, smallAxis), angle, 0, 360, c, 2);
					char id_str[10];
					itoa(i, id_str, 10);
					//Point origin (fr_.bbox.x, fr_.bbox.y+15);
					putText(frame, id_str, Point(fr_.mean.x + bigAxis, fr_.mean.y), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255));
				}

			}
			
			w << frame;
			cout << "Frame id: " << frId << endl;

			imshow("MoA", frame);
			int c = waitKey(wait);
			if (c == 13)
				wait = 1;


			frId++;
		}

	}
	cap.release();
}

void initializeColours()
{
	Scalar c = Scalar::all(-1);
	for (int i = 0; i < 100; i++)
	{
		colorsGT[i] = c;
		colorsST[i] = c;
	}
}


void readAndDrawBboxes()
{
	srand (time(NULL));
	initializeColours();
	////Read gt file generated from vatic (bounding boxes)
	//ifstream inGt (VATIC_GT_FILE);
	////Output where the bounding boxes habe been converted in covariance matrices
	//ofstream outGt (ELLIPSE_GT_FILE);
	//
	//vector<vector<Frame>> gtData;
	////load data
	//readVaticGtFile(inGt, gtData);
	////I need to add this method inside "readVAticGTFile"
	//writeXMLEllipseGT(gtData, outGt);
	////Detect DCFs
	//vector<DCF> DCFs;
	//dcfDetection(DCFs, gtData);
	//writeXMLDCFs(DCFs);

	
	//inGt.close();
	//outGt.close();

	//Read the ground truth ellipses and detect DCFs
//	ifstream inGt2 (ELLIPSE_GT_FILE);
//	vector<vector<FrameEllipse>> gtEllipseData;
//	readEllipseFile(inGt2, gtEllipseData, true);
//	vector<DCF> DCFs;
//	dcfDetectionEllipse(DCFs, gtEllipseData);
//	writeXMLDCFs(DCFs);
//	writeTXTDCFs(DCFs);
//	//
//	readDCFsFile(DCFs);
//	list<Merge> mergeListGt, mergeListPostGt;
//	detectMerges(DCFs, mergeListGt);
//	postProcess(mergeListGt, mergeListPostGt, gtEllipseData); //gtEllipseData is orderd according to targetId
////	drawMerges(mergeListPostGt);
//	writeXMLMerges(mergeListPostGt);
//
//	list<Merge> mergeListSt;
//	readXMLSMerges(mergeListSt, STMERGES_FILE);
//	drawS_GTmerges(mergeListPostGt, mergeListSt);

	//
	//inGt2.close();

	//Draw ellipses from file	
	ifstream inEllipseSt ("d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\Tracks_ellipses.txt");
	//ifstream inEllipseGt (ELLIPSE_GT_FILE);
	vector<vector<FrameEllipse>> stEllipseData1;
	readEllipseFile(inEllipseSt, stEllipseData1, true);
	//vector<DCF> DCFs_II;
	//readDCFsFile(DCFs_II);
	//drawEllipses(gtEllipseData1, DCFs_II);

	//Draw gtruth interaction periods and the people detection from system
	list<Merge> mergeListGt;
	readXMLSMerges(mergeListGt, MERGE_GT_FILE_XML);
	drawGtmerges_stracks(mergeListGt, stEllipseData1);

	//Draw both Gt and Stracks
	/*vector<vector<FrameEllipse>> gtEllipseData, stEllipseData;
	ifstream inGt3 (ELLIPSE_GT_FILE);
	readEllipseFile(inGt3, gtEllipseData, true);
	ifstream inSt ("d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet2\\Tracks_ellipses.txt");
	readEllipseFile(inSt, stEllipseData, false);
	drawST_GTtracks(gtEllipseData, stEllipseData);*/


	//int ttlTgt = gtData.size();
	////draw bbox in sequence
	//VideoCapture cap(MOA_ORIGINAL_FILE);
	//VideoWriter w(MOA_LABELLED_FILE,CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);
	//
	//int frId = 0;
	//int wait = 1;
	//if (cap.isOpened())
	//{
	//	
	//	int idDCF = 0;
	//	while(frId < 1000)
	//	{
	//		cout << "Frame: " << frId << endl;
	//		Mat frame;
	//		cap >> frame;

	//		vector<DCF> dcfsCurrent;
	//		while (idDCF < DCFs.size() && DCFs[idDCF].idFrame == frId)
	//		{
	//			DCF dcf = DCFs[idDCF];
	//			cout << "DCF: " ;
	//			for (int i = 0; i < dcf.trgtIds.size(); i++)
	//			{
	//				cout << dcf.trgtIds[i] << " ";
	//			}
	//			cout << endl;
	//			dcfsCurrent.push_back(dcf);
	//			//wait = 0;
	//			idDCF++;
	//		}
	//

	//		//Draw bboxex
	//		for (int i = 0; i < ttlTgt; i++)
	//		{
	//			vector<Frame> tgt = gtData[i];

	//			bool found = false;
	//			int cont = 0;
	//			Frame fr_;
	//			int ttlFrames = tgt.size();
	//			while (!found && cont < ttlFrames)
	//			{
	//				fr_ = tgt[cont++];
	//				found = (fr_.idFrame >= frId) && (!fr_.out);
	//			}
	//			if (found && fr_.idFrame == frId)
	//			{
	//				Scalar color (0,0,255);
	//				
	//				if (isDCF(fr_, dcfsCurrent))
	//					color = Scalar(0,255,0);

	//				double angle; 
	//				Point center;
	//				int bigAxis = 0; 
	//				int smallAxis = 0;
	//				angle = 0;
	//				double covX, covY, covXY;
	//				getEllipseParams(&frame, fr_.bbox, center, bigAxis, smallAxis, angle, covX, covY, covXY);
	//				//save in file
	//				outGt << i << " " << center.x << " " << center.y << " " << covX << " " << covY << " " << covXY << " " << frId << endl;
	//				ellipse(frame, center, Size(bigAxis, smallAxis), angle, 0, 360, color);
	//				//draw bbox
	//				rectangle(frame, fr_.bbox, color);
	//				char id_str[10];
	//				itoa(i, id_str, 10);
	//				Point origin (fr_.bbox.x, fr_.bbox.y+15);
	//				putText(frame, id_str, origin, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255));
	//			}

	//		}

	//		w << frame;
	//		
	//		imshow("MoA", frame);
	//		waitKey(wait);
	//		if (wait == 0)
	//			wait = 1;
	//		frId++;
	//		
	//	}
	//}

	//cap.release();


}


int main()
{

	readAndDrawBboxes();
	//readAndDrawEllipses();


}
