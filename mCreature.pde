

class mFixture{
  PVector pos=new PVector();
  PVector speed=new PVector();
  float size;
  
  float mess;
  color c;
  
  void update(mFixtureEnv env)
  {
  }
  void preUpdate(mFixtureEnv env)
  {
  }
  void postUpdate(mFixtureEnv env)
  {
  }
  void handleCollideExceedNormal(PVector normalExcced,mFixture collideObj)
  {
  }
  
  void draw(float offsetX,float offsetY)
  {
    fill(c);
    stroke(c);
    ellipse(pos.x+offsetX,-pos.y+offsetY, size, size);
    
    
    fill(255);
    stroke(255);
    line(pos.x+offsetX,-pos.y+offsetY,pos.x+5*speed.x+offsetX,-(pos.y+5*speed.y)+offsetY );
  }
}


  class ConsciousCenter
  {
    float in_energy;
    float in_exhustedLevel;
    float in_currentSpeed;
    float in_peerInfo;
    float in_eyesBeam[]=new float[5];
    
    float inout_mem[]=new float[0];
    
    float ou_turnLeft;
    float ou_turnRight;
    float ou_speedUp;
    float ou_speedDown;
    float ou_pred_eyesBeam[]=new float[in_eyesBeam.length];
    
    s_neuron_net nn = new s_neuron_net(new int[]{4+in_eyesBeam.length+inout_mem.length,10,2,8,4+in_eyesBeam.length+inout_mem.length});
    int histC=0;
    float InX[][]=new float[2][nn.input.length];
    float OuY[][]=new float[InX.length][nn.output.length];
    float predictStateError=1;
    
    
    RLearningCore QL=new RLearningCore(1000, 9, nn.output.length){
       void actExplainX(float q_err[],float q_cx[],float q_nx[],ExpData ed)
      {
        //r(s,a)+garmma*max_a'_(Q_nx) => Q_cx
        //if(ed.R_eward!=0)print(ed.R_eward);
        float garmma=0.98;
        
        int selIdx=(ed.A_ct[0]>ed.A_ct[1])?0:1;
        float maxQ_next_act=(q_nx[0]>q_nx[1])?q_nx[0]:q_nx[1];
        q_err[selIdx]=(ed.R_eward+(garmma)*maxQ_next_act)-q_cx[selIdx];
        q_err[1-selIdx]=0;  
          
        
        selIdx=(ed.A_ct[2]>ed.A_ct[3])?2:3;
        maxQ_next_act=(q_nx[2]>q_nx[3])?q_nx[2]:q_nx[3];
        q_err[selIdx]=(ed.R_eward+(garmma)*maxQ_next_act)-q_cx[selIdx];
        q_err[5-selIdx]=0;  
        
        selIdx = 4;
        /*for(int i=selIdx;i<q_nx.length;i++)//other don't care
          q_err[i]=0;  */
        
        for(int i=0;i<in_eyesBeam.length;i++)
        {
          //Predict next state
          //error = next state - current predict
          q_err[4+i]=ed.S_tate_next[i+selIdx]-q_cx[4+i];
          
          float tmp =q_err[i+selIdx]*500;
          if(tmp<0)tmp=-tmp;
          predictStateError+=0.01*(tmp-predictStateError);
          //q_err[i+selIdx]=0;
        }
          
      };
    };
    
    
    float energy;
  
    ConsciousCenter expShareList[];
    void set_expShareList(ConsciousCenter expShareList[])
    {
      this.expShareList=expShareList;
    }
    
    ConsciousCenter()
    {
      init();
    }
      
    void init()
    {
      in_energy=0;
      in_exhustedLevel=0;
      in_currentSpeed=0;
      in_peerInfo=0;
      for(int i=0;i<in_eyesBeam.length;i++)
      {
        in_eyesBeam[i]=0;
      }
      for(int i=0;i<inout_mem.length;i++)
      {
        inout_mem[i]=0;
      }
    }
    int skipIdx=0;
    
    void histReset()
    {
      skipIdx=0;
      histC=0;
    }
        
    int InoutIdx=0;
    boolean elpsExplore=false;
    void UpdateNeuronInput()
    {
      skipIdx=(skipIdx+1)%1;
      if(skipIdx==0)
      {
        histC++;
        if(histC>InX.length)histC=InX.length;
        InoutIdx++;
        InoutIdx%=InX.length;
      }
      int i=0;
      
      InX[InoutIdx][i++]=in_energy;
      InX[InoutIdx][i++]=in_exhustedLevel;
      InX[InoutIdx][i++]=in_currentSpeed;
      InX[InoutIdx][i++]=in_peerInfo;
      for(int j=0;j<in_eyesBeam.length;j++)
      {
        InX[InoutIdx][i++]=in_eyesBeam[j];
      }
      
      
      for(int j=0;j<inout_mem.length;j++)
      {
        InX[InoutIdx][i++]=inout_mem[j];
      }
      
      
      
      for(int j=0;j<InX[InoutIdx].length;j++)
      {
        nn.input[j].latestVar=InX[InoutIdx][j];
      }
      nn.calc();
      
      for(int j=0;j<OuY[InoutIdx].length;j++)
      {
        OuY[InoutIdx][j]=nn.output[j].latestVar;
      }
      i=0;
      if(elpsExplore)
      {
        if(random(0,1)>0.50)
        {
          OuY[InoutIdx][0]+=(random(0,1)>0.5)?100:-100;
        }
        if(random(0,1)>0.50)
        {
          OuY[InoutIdx][2]+=(random(0,1)>0.5)?100:-100;
        }
      }
      ou_turnLeft=OuY[InoutIdx][i++];
      ou_turnRight=OuY[InoutIdx][i++];
      ou_speedUp=OuY[InoutIdx][i++];
      ou_speedDown=OuY[InoutIdx][i++];
      
      
      
      for(int j=0;j<ou_pred_eyesBeam.length;j++)
      {
        ou_pred_eyesBeam[j]=OuY[InoutIdx][i++];
      }
      
      
      for(int j=0;j<inout_mem.length;j++)
      {
        inout_mem[j]=OuY[InoutIdx][i++];
      }
    }
    

    
    ExpData thisExp=new ExpData(0,0);
    void ReinforcementTraining(float reward,int iter)//+ for reward
    {
      int currentIdx=InoutIdx;
      int prevIdx=currentIdx-1;
      if(prevIdx<0)prevIdx+=InX.length;
      float state[]=InX[prevIdx];
      float act[]=OuY[prevIdx];
      float nstate[]=InX[currentIdx];
      thisExp.ExpLink(state,act,reward,nstate,null);
      
      if(reward!=0||random(0,1)>0.98)
      {
        println(reward);
        QL.pushExp(state,act,reward,nstate,null);
      
        for(int i=0;i<20;i++)
        {
          for(int j=0;j<10;j++)
            QL.RlearningTrainX(nn,QL.expReplaySet[(int)random(0,QL.getAvalibleExpSize())],0.1,false);
          nn.Update_dW(0.1/10);
        }
          
      }
      QL.RlearningTrainX(nn,thisExp,0.1,true);
      
    }
    void BoostingTraining(float alpha)//+ for reward
    {
      int rIdx=InoutIdx;
      for(int i=0;i<InX.length;i++)
      {
          for(int j=0;j<InX[i].length;j++)
          {
            InX[rIdx][j]+=random(-alpha,alpha);
          }
          
          rIdx--;
          if(rIdx<0)rIdx+=InX.length;
      }
      
      
      //training(InX,OuY,1,0.1);
    }

    
  
  }
  

class mFIXStruct extends mFixture
{
  mFIXStruct()
  {
    size=Float.POSITIVE_INFINITY;
    mess=Float.POSITIVE_INFINITY;
  }
}


class mEnergyUp extends mFixture
{
  mEnergyUp()
  {
    size=8;
    mess=0.1;
  }
}
  
  
class mCreature extends mFixture{
  
  mCreature()
  {
    reset();
  }
  
  void reset()
  {
    size=15;
    mess=1;
    pos.x=random(-300,300);
    pos.y=random(-300,300);
    speed.x=random(-2,2);
    speed.y=random(-2,2);
    c=color(random(0,255),random(0,255),random(0,255),100);
    CC.init();
    CC.in_energy=1;
    
  }

  ConsciousCenter CC=new ConsciousCenter();
  
  void handleCollideExceedNormal(PVector normalExcced,mFixture collideObj)
  {
    float crashLevel=normalExcced.mag();
    if((collideObj instanceof mCreature) )
    {
      mCreature cobj=(mCreature)collideObj;
      if(cobj.lifeTime<5)return;
      Reward=-0.4;
    }
    else
    {
      Reward=-0.8;
      lifeTime=0;
      pos.x=random(-200,200);
      pos.y=random(-200,200);
      speed.x=random(-1,1);
      speed.y=random(-1,1);
    }
    
  }
  
  
  protected void rotation_speed(float d)
  {
    float x,y;
    float cos=cos(d),sin=sin(d);
    x= cos*speed.x-sin*speed.y;
    y= sin*speed.x+cos*speed.y;
    
    speed.x=x;
    speed.y=y;
  }
  boolean guideGate=false;
  //HistDataDraw turnHist=new HistDataDraw(1500);
  //HistDataDraw speedHist=new HistDataDraw(1500);
  int speedLowC=0;
  
  PVector prePos=new PVector();
  
  float turnAcc = 0;
  float speedAbs;
  float turnAmount=0;
  int rewardC=0;
  
  float lifeTime=0;
  boolean isfellBad=false;
  mFixtureEnv env=null;
  float eye_spreadAngle=15;
  
  float Reward=0;
  mFixture retCollide[]=new mFixture[1];
  void update(mFixtureEnv env)
  {
     this.env=env;
    CC.in_peerInfo*=0.6;
    float velocity=prePos.dist(pos);
    prePos.set(pos);
      
    
    PVector ret_intersect=new PVector();
    float speedAngle=atan2(speed.y,speed.x)-eye_spreadAngle*PI/180*(CC.in_eyesBeam.length-1)/2;
    
    float minDist=Float.POSITIVE_INFINITY;
    float maxDist=0;
    for(int i=0;i<CC.in_eyesBeam.length;i++)
    {
      
      float distret=env.testBeamCollide(pos,speedAngle+eye_spreadAngle*PI/180*i,ret_intersect, retCollide);
      if(minDist>distret)minDist=distret;
      if(maxDist<distret)maxDist=distret;
      if(distret>300)distret=300;
      CC.in_eyesBeam[i]=distret/500;
      if(retCollide[0] instanceof mCreature)
      {
        mCreature collideCre = (mCreature)retCollide[0];
        //Reward+=0.03;
        //collideCre.CC.in_peerInfo+=CC.ou_expectReward;
      }
      else
      {
      }

    }
    //println("minDist="+minDist);
   // minDist=(minDist+maxDist)/2;
    if(lifeTime>500&&minDist>100)
    {
      lifeTime=0;
      Reward+=0.2;
    }
      
    CC.UpdateNeuronInput();
    
    CC.in_energy*=0.9999;
    
   
    
    
    //stroke(0,255,0,100);
    //turnHist.Draw(CC.ou_turnSpeed*10,0,300,width,500);
    if(CC.ou_speedUp>CC.ou_speedDown)
    {
      speed.mult(1.1);
    }
    else
    {
      speed.mult(1/1.1);
    }
    
    //speed.mult(CC.ou_speedAdj);
    speedAbs=speed.mag();
    CC.in_currentSpeed=speedAbs/3;
    
    float turn=CC.ou_turnLeft>CC.ou_turnRight?1:-1;
    turn*=1.5;
    rotation_speed(turn*PI/180);
    turnAmount+=turn;
    
    if(turnAmount>500||turnAmount<-500)
    {
      turnAmount=0;
      
      lifeTime=0;
      Reward+=-0.2;
    }
    //stroke(128,200,0,100);
    //speedHist.Draw(CC.ou_speedAdj*10,0,300,width,500);
    
    
    if(speedAbs>3)
    {
      speed.mult(0.9);
    }
    else if(speedAbs<0.5)
      speed.mult(random(1.1,1.2));

    lifeTime+=speedAbs/5+0.5;
    
    pos.add(speed);
  }
  
  void preUpdate(mFixtureEnv env)
  {
    Reward=0;
  }
  void postUpdate(mFixtureEnv env)
  {
    CC.nn.PreTrainProcess(0.01);
    CC.ReinforcementTraining(Reward,1);
    Reward=0;
  }
  void draw(float offsetX,float offsetY)
  {
    
    stroke(c,100);
    fill(c,50);
    
    PVector ret_intersect=new PVector();
    float speedAngle=atan2(speed.y,speed.x)-eye_spreadAngle*PI/180*(CC.in_eyesBeam.length-1)/2;
    
    if(env!=null)
    for(int i=0;i<CC.in_eyesBeam.length;i++)
    {
      
      env.testBeamCollide(pos,speedAngle+eye_spreadAngle*PI/180*i,ret_intersect, retCollide);
      //ellipse(ret_intersect.x+env.frameW/2,-ret_intersect.y+env.frameH/2, 15, 15);
      line(ret_intersect.x+env.frameW/2,-ret_intersect.y+env.frameH/2,pos.x+env.frameW/2,-pos.y+env.frameH/2);

    }
    

    for(int i=0;i<CC.in_eyesBeam.length;i++)
    {
      
      float ang=speedAngle+eye_spreadAngle*PI/180*i;
      float predictDist=CC.in_eyesBeam[i]*500;
      stroke(c,50);
      ellipse(predictDist*cos(ang)+pos.x+env.frameW/2,-predictDist*sin(ang)-pos.y+env.frameH/2, 15, 15);
      
      
      predictDist=CC.ou_pred_eyesBeam[i]*500;
      
      stroke(c,255);
      ellipse(predictDist*cos(ang)+pos.x+env.frameW/2,-predictDist*sin(ang)-pos.y+env.frameH/2, 15, 15);
    }
    
    fill(c);
    stroke(c);
    
    ellipse(pos.x+offsetX,-pos.y+offsetY, size, size);
    
    
    fill(255);
    stroke(255);
    line(pos.x+offsetX,-pos.y+offsetY,pos.x+5*speed.x+offsetX,-(pos.y+5*speed.y)+offsetY );
    
    noFill();
    
    
  }
  
}


  
class mCreatureEv extends mCreature implements Comparable<mCreatureEv>{
  
  int lifeTime=0;
  float turnX=0;
  float speeUpC=0;
  int seeOtherC=0;
  int HitMark=0;
  
  public int compareTo(mCreatureEv other) {
      return Float.compare(other.getFitness(),getFitness());// name.compareTo(other.name);
  }
  
  
  void clone(mCreatureEv from)
  {
    
    s_neuron_net nn_from[]=new s_neuron_net[1];
    float fitness[] =new float[1];
    nn_from[0]=from.CC.nn;
    fitness[0]=from.getFitness();
    CC.nn.GeneticCrossNN(nn_from,fitness);
    revive();
    
    turnX=from.turnX;
    speeUpC=from.speeUpC;
    lifeTime=from.lifeTime;
    seeOtherC=from.seeOtherC;
    
  }
  
  mCreatureEv()
  {
    super();
    revive();
  }
  mCreatureEv(mCreatureEv parents[],float fitness[])
  {
    this();
    birth(parents,fitness);
  }
  
  mCreatureEv(mCreatureEv cloneFrom)
  {
    this();
    clone(cloneFrom);
  }
  void revive()
  {
    reset();
    lifeTime=0;
    turnX=0;
    seeOtherC=0;
    speeUpC=0;
    CC.in_energy=0.5;
    size=10;
    HitMark=0;
  }
  
  void birth(mCreatureEv parents[],float fitness[])
  {
    s_neuron_net nn_parents[]=new s_neuron_net[parents.length];
    for(int i=0;i<nn_parents.length;i++)
    {
      nn_parents[i]=parents[i].CC.nn;
    }
    CC.nn.GeneticCrossNN(nn_parents,fitness);
    if(random(0,1)>0.5)CC.nn.AddNNNoise(0.03);
    if(random(0,1)>0.7)CC.nn.AddNNmutate(0.1);
    CC.nn.PreTrainProcess(0.01);
    revive();
    
  }
  
  
  ConsciousCenter CC=new ConsciousCenter();
  
  void handleCollideExceedNormal(PVector normalExcced,mFixture collideObj)
  {
    float mag=normalExcced.mag();
    if((collideObj instanceof mCreatureEv) )
    {
      CC.in_energy-=0.002*mag;
      HitMark+=20;
    }
    else
    {
      CC.in_energy-=0.03*mag;
      HitMark=255;
    }
      
  }
  
  float turnAcc = 0;
  float speedAbs;
  
  float getFitness()
  {
    float turnAmount=turnX>0?turnX:-turnX;
    return lifeTime-turnAmount*2+speeUpC/4;
  }
  
  boolean isfellBad=false;
  void update(mFixtureEnv env)
  {
    float spreadAngle=100/CC.in_eyesBeam.length;
    PVector ret_intersect=new PVector();
    float speedAngle=atan2(speed.y,speed.x)-spreadAngle*PI/180*(CC.in_eyesBeam.length-1)/2;
    
    float minDist=Float.POSITIVE_INFINITY;
    float maxDist=0;
    stroke(c,100);
    fill(c,100);
    
    mFixture retCollide[]=new mFixture[1];
    for(int i=0;i<CC.in_eyesBeam.length;i++)
    {
      
      float distret=env.testBeamCollide(pos,speedAngle+spreadAngle*PI/180*i,ret_intersect, retCollide);
      if(minDist>distret)minDist=distret;
      if(maxDist<distret)maxDist=distret;
     
      //ellipse(ret_intersect.x+env.frameW/2,-ret_intersect.y+env.frameH/2, 15, 15);
      //line(ret_intersect.x+env.frameW/2,-ret_intersect.y+env.frameH/2,pos.x+env.frameW/2,-pos.y+env.frameH/2);
      if(retCollide[0] instanceof mCreatureEv)
      {
        mCreatureEv collideCre = (mCreatureEv)retCollide[0];
        
        //collideCre.CC.in_peerInfo+=CC.ou_expectReward/distret;
        
        //collideCre.CC.in_energy+=0.001/distret;
        CC.in_energy+=0.01/distret/CC.in_eyesBeam.length;
        CC.in_eyesBeam[i]=distret/1000;
      }
      else
      {
         CC.in_eyesBeam[i]=distret/1000;
      }

    
    }
   // CC.in_energy*=0.95;
    //if(CC.in_energy>1)CC.in_energy=1;
    CC.in_peerInfo*=0.9;
    CC.UpdateNeuronInput();
    
    
    
     
    //stroke(0,255,0,100);
    //turnHist.Draw(CC.ou_turnSpeed*10,0,300,width,500);
    if(CC.ou_speedUp>CC.ou_speedDown)
    {
      speed.mult(1.1);
    }
    else
    {
      speed.mult(1/1.1);
    }
    
    //speed.mult(CC.ou_speedAdj);
    speedAbs=speed.mag();
    CC.in_currentSpeed=speedAbs/2;
    
    float turn=CC.ou_turnLeft>CC.ou_turnRight?1:-1;
    turn*=1.5;
    
    
    //speed.mult(CC.ou_speedAdj);
    speedAbs=speed.mag();
    //CC.in_energy-=0.001/speedAbs;
    CC.in_currentSpeed=speedAbs/2;
    rotation_speed(turn*PI/180);
    turnX+=turn;
    
    {
      float absTurnX=(turnX>0)?turnX:-turnX;
      CC.in_energy-=0.0000001*absTurnX;
    }
    //stroke(128,200,0,100);
    //speedHist.Draw(CC.ou_speedAdj*10,0,300,width,500);
    
    if(CC.in_energy>0.8&&random(0,1)>0.8&&HitMark==0){
      //println("Hit:"+preFiness);
      CC.BoostingTraining(0.2);
    }
   
      
    if(speedAbs>3)
    {
      speed.mult(0.9);
    }
    else if(speedAbs<0.2)
      speed.mult(1.1);

    lifeTime+=1;//speed.mag()/5+0.5;
    pos.add(speed);
  }
  void draw(float offsetX,float offsetY)
  {
    fill(c);
    stroke(c);
    ellipse(pos.x+offsetX,-pos.y+offsetY, size, size);
    
    
    fill(255);
    stroke(255);
    line(pos.x+offsetX,-pos.y+offsetY,pos.x+5*speed.x+offsetX,-(pos.y+5*speed.y)+offsetY );
    
    noFill();
    
    
    if(HitMark>0)
    {
      stroke(255,50,150);
      
    }
    HitMark=HitMark*13/14;
    arc(pos.x+offsetX,-pos.y+offsetY,size+2+HitMark/10, size+2+HitMark/10, 0, 2*CC.in_energy*PI);
  }
  
}