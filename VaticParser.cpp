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

struct DCF
{
	int idFrame;
	vector<int> trgtIds;
};


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

int main()
{
	ifstream inGt ("d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\gt.txt");
	string line;
	vector<vector<Frame>> gtData;

	int idTgtPrev = -1;
	//load data
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
	int ttlTgt = gtData.size();
	//detect DCFs
	vector<DCF> DCFs;
	//go through all frames
	int* tgtCount = new int[ttlTgt];
	initCounters(tgtCount, ttlTgt);
	for (int i = 0; i < 1000; i++)
	{
		//cout << "Frame: " << i << endl
		vector<Frame> tgtFrames; //stores the frames of the targets that are present in the current frame
		for (int j = 0; j < ttlTgt; j++)
		{
			int idCount = tgtCount[j];
			if (idCount >= gtData[j].size())
				continue;

			Frame f = gtData[j][idCount];
			if (f.idFrame == i)
			{
				if (!f.out)
					tgtFrames.push_back(f); //position of targets that are present in the current frame
					
				tgtCount[j]++; //Position of the list of frames of the target.
			}
		}
	
		//go through all targets present in the current frame
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
				}
			}
		}
	}

	

	//draw bbox in sequence
	VideoCapture cap("d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\MoA_Detection_final.mpg");
	VideoWriter w("d:\\Emilio\\Tracking\\DataSet\\sb125\\SecondDay\\DSet1\\outputDCF.mpg",CV_FOURCC('P','I','M','1'), 20.0, Size(1148,480), true);
	
	int frId = 0;
	int wait = 1;
	if (cap.isOpened())
	{
		
		int idDCF = 0;
		while(frId < 1000)
		{
			cout << "Frame: " << frId << endl;
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
					Scalar color (0,0,255);
					
					if (isDCF(fr_, dcfsCurrent))
						color = Scalar(0,255,0);
					
					//draw bbox
					rectangle(frame, fr_.bbox, color);
					char id_str[10];
					itoa(i, id_str, 10);
					Point origin (fr_.bbox.x, fr_.bbox.y+15);
					putText(frame, id_str, origin, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, Scalar(0,0,255));
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
