float InArr[][]=new float[100][2];//x,y=2
float OuArr[][]=new float[InArr.length][2];//output class1,2
float error[][]=new float[InArr.length][2];//output class1,2

NeuralUtil nu=new NeuralUtil();
neuroLayer L1 = new neuroLayer(nu,InArr.length,InArr[0].length,OuArr[0].length);
void setup() {
  size(640, 360);
  for (int i=0; i<InArr.length; i++) {
    float offset = (i%2==0)? 1:-1;
    offset*=0.9;
    InArr[i][0]=offset+random(-1, 1);//x
    InArr[i][1]=random(-1, 1);//y
    OuArr[i][(i%2==0)?1:0]=1;//class 2
  }
  frameRate(30);
}

void draw(){
  background(0);
  
  scanPlain();
  //Run
  L1.ForwardPass(InArr);
  
  
  
  drawX(InArr,OuArr,L1.pred_Y);
  //Train
  nu.matAdd(error,OuArr,L1.pred_Y,-1);
  L1.backProp(null,error);
  L1.updateW(0.01);
  L1.reset_deltaW();
  //
}
 
void drawX(float in[][],float out[][],float pred[][]) {
  float mult=100;
  for (int i=0; i<in.length; i++) {
    if (out[i][0]==1)
    {
      fill(255, 0, 0);
      stroke(255, 0, 0);
    }
    else
    {
      fill(0, 255, 0);
      stroke(0,255, 0);
    }
    ellipse(width/2+in[i][0]*mult, height/2+in[i][1]*mult, 8, 8);
    
    noFill();
    if (pred[i][0]>pred[i][1])
      stroke(255, 100, 100);
    else
      stroke(100, 255, 100);
    ellipse(width/2+in[i][0]*mult, height/2+in[i][1]*mult, 10, 10);
  }
}

void drawScanPlain(float in[][],float pred[][]) {
  float mult=100;
  for (int i=0; i<in.length; i++) {
    
    noFill();
    if (pred[i][0]>pred[i][1])
      stroke(255, 100, 100,50);
    else
      stroke(100, 255, 100,50);
    ellipse(width/2+in[i][0]*mult, height/2+in[i][1]*mult, 10, 10);
  }
}
float scan1[][]=new float[InArr.length][InArr[0].length];
void scanPlain()
{
  float x=-1,y=-1;
  while(true)
  {
    for (int i=0; i<scan1.length; i++) {
      scan1[i][0]=x;//x
      scan1[i][1]=y;//y
      x+=0.1;
      if(x>1)
      {
        x=-1;
        y+=0.1;
      }
    }
      
    L1.ForwardPass(scan1);
    drawScanPlain(scan1,L1.pred_Y);
    
    if(y>1)break;
    
  }
}


class neuroLayer {
  float InArr[][];
  float pred_preY[][];
  float pred_Y[][];
  float error_gradient[][];
  float W[][],dW[][];

  NeuralUtil nu;
  neuroLayer(NeuralUtil nu,int batchSize,int inDim,int ouDim)
  {
    this.nu = nu;
    InArr=new float[batchSize][inDim+1];
    pred_preY=new float[batchSize][ouDim];
    pred_Y=new float[batchSize][ouDim];
    error_gradient=new float[batchSize][ouDim];
    W=nu.genWMat(InArr,ouDim);
    dW=nu.genWMat(InArr,ouDim);
    
    for(int i=0;i<InArr.length;i++)
    {
      InArr[i][InArr[i].length-1]=1;
    }
  }
  neuroLayer(NeuralUtil nu,neuroLayer preLayer,int layerDim)
  {
    this(nu,preLayer.InArr.length,preLayer.InArr[0].length-1,layerDim);
  }
  
  void ForwardPass(float in[][])
  {
    for(int i=0;i<in.length;i++)
      for(int j=0;j<in[i].length;j++)
    {
      InArr[i][j]=in[i][j];
    }
    nu.matMul(pred_preY,InArr,W);
    
    nu.actvationF(pred_Y,pred_preY);
  }
  void reset_deltaW()
  {
    nu.matZero(dW);
  }
  
  void updateW(float learningRate)
  {
    nu.matAdd(W,W,dW,learningRate/pred_Y.length);
  }
  
  void backProp(float back_gradient[][],float error_gradient[][])
  {
    
    nu.gradient_actvationF(this.error_gradient,pred_preY);//get sigmoid gradient
    
    
    for(int i=0;i<error_gradient.length;i++)//Multiply error gradient with sigmoid gradient
      for(int j=0;j<error_gradient[0].length;j++)
    {
      this.error_gradient[i][j]*=error_gradient[i][j];
    }
    
    nu.deltaW_accumulate(dW,InArr,this.error_gradient);
    if(back_gradient!=null)
      nu.backgradient(back_gradient,W,this.error_gradient);
  }
  
}
class NeuralUtil{
  float [][] genWMat(float [][] input,int layerSize){
    float[][] wRandom=new float[input[0].length][layerSize];
    for (int i = 0; i < wRandom.length; i++) { // aRow
          for (int j = 0; j < wRandom[0].length; j++) { // bColumn
            wRandom[i][j]=random(-1,1)/10;
          }
      }
    return wRandom;
  }
   
  
  //AB  EF
  //CD  GH 
  //AE+BG , AF+BH
  //CE+DG , CF+DH
  void matMul(float[][] C,float[][] A, float[][] B) {
        int aRows = A.length;
        int aColumns = A[0].length;
        int bRows = B.length;
        int bColumns = B[0].length;

        if (aColumns != bRows) {
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
        }
        for (int i = 0; i < C.length; i++) {
            for (int j = 0; j < C[0].length; j++) {
                C[i][j] = 0.0f;
            }
        }
        for (int i = 0; i < aRows; i++) { // aRow
            for (int j = 0; j < bColumns; j++) { // bColumn
                for (int k = 0; k < aColumns; k++) { // aColumn
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

    }
    
      
      
  void matAdd(float[][] C,float[][] A, float[][] B) {
      for (int i = 0; i < A.length; i++) { // aRow
          for (int j = 0; j < A[0].length; j++) { // bColumn
            C[i][j]=A[i][j]+B[i][j];
          }
      }
  }
  
  void matAdd(float[][] C,float[][] A, float[][] B,float B_coeff) {
      for (int i = 0; i < A.length; i++) { // aRow
          for (int j = 0; j < A[0].length; j++) { // bColumn
            C[i][j]=A[i][j]+B_coeff*B[i][j];
          }
      }
  }
  
  void matZero(float[][] mat) {
      for (int i = 0; i < mat.length; i++) { // aRow
          for (int j = 0; j < mat[0].length; j++) { // bColumn
            mat[i][j]=0;
          }
      }
  }
  void deltaW_accumulate(float[][] deltaW,float[][] in, float[][] error_gradient) {
            
    for(int k=0;k<in.length;k++)//Iterate each data points
      for (int i = 0; i < deltaW.length; i++) { // aRow
          for (int j = 0; j < deltaW[0].length; j++) { // bColumn
              deltaW[i][j]+=error_gradient[k][j] * in[k][i];
          }
      }
  }
  void backgradient(float[][] backg,float[][] W, float[][] error_gradient) {
            
    for (int i = 0; i < backg.length; i++) {
        for (int j = 0; j < backg[0].length; j++) {
            backg[i][j]=0;
        }
    } 
    for (int k = 0; k < backg.length; k++) {
        for (int i = 0; i < backg[0].length; i++) {
          for(int j=0;j<W[0].length;j++)
            backg[k][i]+=error_gradient[k][j] * W[i][j];
        }
    }
  }
  void leRu(float[][] out,float[][] in){
     for (int i = 0; i < in.length; i++) { // aRow
          for (int j = 0; j < in[0].length; j++) { // bColumn
            out[i][j]=in[i][j]>0?in[i][j]:0;
          }
      }
  }
  
  void gradient_leRu(float[][] out,float[][] in){
     for (int i = 0; i < in.length; i++) { // aRow
          for (int j = 0; j < in[0].length; j++) { // bColumn
            out[i][j]=in[i][j]/(1-in[i][j]);
          }
      }
  }
  
  void sigmoid(float[][] out,float[][] in){
     for (int i = 0; i < in.length; i++) { // aRow
          for (int j = 0; j < in[0].length; j++) { // bColumn
            out[i][j]=1/(1+exp(-in[i][j]));
          }
      }
  }
  
  void actvationF(float[][] out,float[][] in)
  {
     sigmoid(out,in);
  }

  void gradient_actvationF(float[][] out,float[][] in){
    
    sigmoid(out,in);
    gradient_sigmoid_Y(out,out);
  }

  void gradient_sigmoid_Y(float[][] out,float[][] in){
     for (int i = 0; i < in.length; i++) { // aRow
          for (int j = 0; j < in[0].length; j++) { // bColumn
            out[i][j]=in[i][j]*(1-in[i][j]);
          }
      }
  }

  public void printMat(float [][]C)
  {
      for (int i = 0; i < C.length; i++) { // aRow
          for (int j = 0; j < C[0].length; j++) { // bColumn
            print(C[i][j]+",");
          }
          println();
      }
  }
}

class NeuralUtil_Linear extends NeuralUtil{
  
  void actvationF(float[][] out,float[][] in)
  {
      for (int i = 0; i < out.length; i++) { // aRow
          for (int j = 0; j < out[0].length; j++) { // bColumn
            out[i][j]=in[i][j];
          }
      }//out = in
  }

  void gradient_actvationF(float[][] out,float[][] in){
      for (int i = 0; i < out.length; i++) { // aRow
          for (int j = 0; j < out[0].length; j++) { // bColumn
            out[i][j]=1;
          }
      }//gradient = 1
  }
}