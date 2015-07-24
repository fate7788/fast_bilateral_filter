#include <opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int main()
{
	double startTime=getTickCount();
	Mat src=imread("FOV00002-28.bmp",1);
	if (!src.data)
	{
		return 0;
	}

	Mat dst=Mat::zeros(src.size(),src.type());
	//算法参数
	int d=9;
	double sigma_color=40.0;
	double sigma_space=2.0;
	int borderType=BORDER_REFLECT;


	int cn = src.channels();
	int  maxk, radius;
	Size size = src.size();

	CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3);

	if( sigma_color <= 0 )
		sigma_color = 1;
	if( sigma_space <= 0 )
		sigma_space = 1;

	double gauss_color_coeff = -0.5/(sigma_color*sigma_color);
	double gauss_space_coeff = -0.5/(sigma_space*sigma_space);

	if( d <= 0 )
		radius = cvRound(sigma_space*1.5);
	else
		radius = d/2;
	radius = MAX(radius, 1);
	d = radius*2 + 1;

	Mat temp;
	copyMakeBorder( src, temp, radius, radius, radius, radius, borderType );

	//颜色、空间权重
	vector<float> _color_weight(cn*256);
	vector<float> _space_weight(d);
	vector<int> _space_of_horizontal(d);//水平
	vector<int> _space_of_vertical(d);//垂直
	float* color_weight = &_color_weight[0];
	float* space_weight = &_space_weight[0];
	int* space_of_horizontal = &_space_of_horizontal[0];
	int* space_of_vertical=&_space_of_vertical[0];

	// initialize color-related bilateral filter coefficients
	int i;
	for( i = 0; i < 256*cn; i++ )
		color_weight[i] = (float)std::exp(i*i*gauss_color_coeff);

	// initialize space-related bilateral filter coefficients
	for ( i=-radius,maxk=0;i<=radius;i++)
	{
		double r=std::abs(radius);
		space_weight[maxk]=(float)std::exp(r*r*gauss_space_coeff);
		space_of_horizontal[maxk]=(int)(i*cn);
		space_of_vertical[maxk++]=(int)(i*temp.step);

	}

	//////////////////////////////////////////////////////////////////////////
	//水平方向双边滤波
	for (int h=0;h<src.rows;h++)
	{
		const uchar* sptr=temp.ptr(h+radius)+radius*cn;
		uchar* dptr=dst.ptr(h);
		if (cn==1)
		{
			for (int w=0;w<src.cols;w++)
			{
				float sum=0,wsum=0;
				int val0=sptr[w];
				int k=0;
				for (;k<maxk;k++)
				{
					int val=sptr[w+space_of_horizontal[k]];
					float w=space_weight[k]*color_weight[abs(val-val0)];
					sum+=val*w;
					wsum+=w;
				}
				dptr[w]=(uchar)cvRound(sum/wsum);
			}
		}
		else 
		{
			assert(cn==3);
			for (int w=0;w<src.cols*3;w++)
			{
				float sumb=0,sumg=0,sumr=0,wsum=0;
				int b0=sptr[w],g0=sptr[w+1],r0=sptr[w+2];
				int k=0;
				for (;k<maxk;k++)
				{
					const uchar* sptr_k=sptr+w+space_of_horizontal[k];
					int b=sptr_k[0],g=sptr_k[1],r=sptr_k[2];
					float w=space_weight[k]*color_weight[abs(b-b0)+abs(g-g0)+abs(r-r0)];
					sumb+=b*w;sumg+=g*w;sumr+=r*w;
					wsum+=w;
				}

				wsum=1.0f/wsum;
				b0 = cvRound(sumb*wsum);
				g0 = cvRound(sumg*wsum);
				r0 = cvRound(sumr*wsum);
				dptr[w] = (uchar)b0; dptr[w+1] = (uchar)g0; dptr[w+2] = (uchar)r0;
			}
		}
		
	}

	imwrite("horizontal.bmp",dst);
	//////////////////////////////////////////////////////////////////////////
	//垂直方向双边滤波
	Mat temp2;
	copyMakeBorder( dst, temp2, radius, radius, radius, radius, borderType );

	for (int h=0;h<src.rows;h++)
	{
		const uchar* sptr=temp2.ptr(h+radius)+radius*cn;
		uchar* dptr=dst.ptr(h);
		if (cn==1)
		{
			for (int w=0;w<src.cols;w++)
			{
				float sum=0,wsum=0;
				int val0=sptr[w];
				int k=0;
				for (;k<maxk;k++)
				{
					int val=sptr[w+space_of_vertical[k]];
					float w=space_weight[k]*color_weight[abs(val-val0)];
					sum+=val*w;
					wsum+=w;
				}
				dptr[w]=(uchar)cvRound(sum/wsum);
			}
		}
		else
		{
			assert(cn==3);
			for (int w=0;w<src.cols*3;w++)
			{
				float sumb=0,sumg=0,sumr=0,wsum=0;
				int b0=sptr[w],g0=sptr[w+1],r0=sptr[w+2];
				int k=0;
				for (;k<maxk;k++)
				{
					const uchar* sptr_k=sptr+w+space_of_vertical[k];
					int b=sptr_k[0],g=sptr_k[1],r=sptr_k[2];
					float w=space_weight[k]*color_weight[abs(b-b0)+abs(g-g0)+abs(r-r0)];
					sumb+=b*w;sumg+=g*w;sumr+=r*w;
					wsum+=w;
				}

				wsum=1.0f/wsum;
				b0 = cvRound(sumb*wsum);
				g0 = cvRound(sumg*wsum);
				r0 = cvRound(sumr*wsum);
				dptr[w] = (uchar)b0; dptr[w+1] = (uchar)g0; dptr[w+2] = (uchar)r0;
			}
		}
	}

	imwrite("vertical.bmp",dst);
	double during=((double)getTickCount()-startTime)/(double)getTickFrequency();

	
	return 1;
}