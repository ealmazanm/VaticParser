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

#include <tinystr.h>
#include <tinyxml.h>

using namespace std;
using namespace cv;


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

struct DCF
{
	int idFrame;
	vector<int> trgtIds;
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

void getEllipseParams(int& bigAxis, int& smallAxis, double& angle, const FrameEllipse* fr)
{
	float covValues[] = {fr->covX, fr->covXY, fr->covXY, fr->covY};
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

const char* MOA_ORIGINAL_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\MoA_Detection_final.mpg";
const char* MOA_LABELLED_FILE = "d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputDCF.mpg";


void readEllipseGtFile(ifstream& inEllipseGt, vector<vector<FrameEllipse>>&gtEllipseData)
{
	string line;
	int idTgtPrev = -1;
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

void drawEllipses(vector<vector<FrameEllipse>>& gtData, vector<DCF>& DCFs)
{
	VideoWriter w(MOA_LABELLED_FILE,CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);

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
				cout << "DCF: " ;
				for (int i = 0; i < dcf.trgtIds.size(); i++)
				{
					cout << dcf.trgtIds[i] << " ";
				}
				cout << endl;
				dcfsCurrent.push_back(dcf);
				wait = 0;
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
					Scalar color (0,0,255);
					
					if (isDCF(fr_, dcfsCurrent))
						color = Scalar(0,255,0);

					double angle;
					int bigAxis = 0; 
					int smallAxis = 0;
					angle = 0;
					double covX, covY, covXY;
					getEllipseParams(bigAxis, smallAxis, angle, &fr_);
					
					ellipse(frame, fr_.mean, Size(bigAxis, smallAxis), angle, 0, 360, color);
					char id_str[10];
					itoa(i, id_str, 10);
					//Point origin (fr_.bbox.x, fr_.bbox.y+15);
					putText(frame, id_str, fr_.mean, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255));
				}

			}

			w << frame;
			
			imshow("MoA", frame);
			waitKey(1);
			if (wait == 0)
				wait = 1;
	


			frId++;
		}

	}
	cap.release();
}


void readAndDrawBboxes()
{
	//Read gt file generated from vatic (bounding boxes)
	ifstream inGt (VATIC_GT_FILE);
	//Output where the bounding boxes habe been converted in covariance matrices
	ofstream outGt (ELLIPSE_GT_FILE);
	
	vector<vector<Frame>> gtData;
	//load data
	readVaticGtFile(inGt, gtData);
	//I need to add this method inside "readVAticGTFile"
	writeXMLEllipseGT(gtData, outGt);
	//Detect DCFs
	vector<DCF> DCFs;
	dcfDetection(DCFs, gtData);
	writeXMLDCFs(DCFs);

	inGt.close();
	outGt.close();

	//Draw ellipses from file
	ifstream inEllipseGt (ELLIPSE_GT_FILE);
	vector<vector<FrameEllipse>> gtEllipseData;
	readEllipseGtFile(inEllipseGt, gtEllipseData);
	vector<DCF> DCFs_II;
	readDCFsFile(DCFs_II);
	drawEllipses(gtEllipseData, DCFs_II);




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
