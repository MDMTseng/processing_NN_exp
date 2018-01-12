

class HistDataDraw{
  
  float Hist[];
  HistDataDraw(int L)
  {
    Hist=new float[L];
  }
  int HistH=0;
  void Draw(float newData,int x,int y,int w,int h) {
    
    DataPush(newData);
    Draw( x, y, w, h);
  }
  
  void Draw(int x,int y,int w,int h) {
    
    line(x,y+h,x+w,y+h);
    int PreadH=0;
    int readH=HistH;
    for(int i=1;i<Hist.length-1;i++)
    {
      PreadH=readH;
      readH=(readH+1)%Hist.length;
      
      line(i*w/Hist.length+x,(h-Hist[PreadH])+y,
      (i+1)*w/Hist.length+x,(h-Hist[readH])+y);
    }
  }
  void DataPush(float newData) {
    Hist[HistH]=newData;
    HistH=(HistH+1)%Hist.length;
  }
}

class DataFeedDraw{
  float PreData;
  void Draw(float newData,int idx,int idxTop,int x,int y,int w,int h) {
    line((idx-1)*w/idxTop+x,(h-PreData)+y,
         (idx)*w/idxTop+x,  (h-newData)+y);
    PreData=newData;
  }
}
class DataFeedDraw2D{
  float PreDataX;
  float PreDataY;
  void reset ()
  {
    PreDataX = 0;
    PreDataY = 0;
    
  }
  void Draw(float newDataX,float newDataY,int x,int y,int w,int h) {
    line((newDataX)*w+x,(h-PreDataY)+y,
         (newDataX)*w+x,  (h-newDataY)+y);
    PreDataX=newDataX;
    PreDataY=newDataY;
  }
}