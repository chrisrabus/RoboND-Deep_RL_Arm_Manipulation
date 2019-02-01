/* 
 * Author - Dustin Franklin (Nvidia Jetson Developer)
 * Modified by - Sahil Juneja, Kyle Stewart-Frantz
 *
 */

#include "ArmPlugin.h"
#include "PropPlugin.h"


#include "cudaMappedMemory.h"
#include "cudaPlanar.h"

#define PI 3.141592653589793238462643383279502884197169f

#define JOINT_MIN	-0.75f //
#define JOINT_MAX	 2.00f //

// Turn on velocity based control
#define VELOCITY_CONTROL false
#define VELOCITY_MIN -0.2f
#define VELOCITY_MAX  0.2f

// Define DQN API Settings

#define INPUT_CHANNELS 3   // 3-channels of RGB image
#define ALLOW_RANDOM true
#define DEBUG_DQN false
#define GAMMA 0.9f
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 200

/*
/ TODO - Tune the following hyperparameters ->DONE
/
*/

#define NUM_ACTIONS DOF*2   //for each active joint, there are 2 actions -> increase/decrease
#define INPUT_WIDTH   64 	//width of camera image
#define INPUT_HEIGHT  64 	//height of camera image
#define OPTIMIZER "RMSprop"
#define LEARNING_RATE 0.1f //0.01f
#define REPLAY_MEMORY 20000
#define BATCH_SIZE 512
#define USE_LSTM true
#define LSTM_SIZE 512

/*
/ TODO - Define Reward Parameters ->DONE
/
*/

// Define max Distances of Gripper from Object
#define MAX_DIST_1 2.43f	//2.43f
#define MAX_DIST_2 2.63f	//2.52f

#define REWARD_WIN  300.0f //1.0f 
#define REWARD_LOSS -300.0f //-1.0f
#define REWARD_INTERIM 200.0f
#define ALPHA    0.9f //0.3f for task 1

// Define Object Names
#define WORLD_NAME "arm_world"
#define PROP_NAME  "tube"
#define GRIP_NAME  "gripper_middle"
#define JOINT1_NAME "joint1"
#define JOINT2_NAME "joint2"

// Define Collision Parameters
#define COLLISION_FILTER "ground_plane::link::collision"
#define COLLISION_ITEM   "tube::tube_link::tube_collision"
#define COLLISION_POINT  "arm::gripperbase::gripper_link"

// Animation Steps
#define ANIMATION_STEPS 1000

// Set Debug Mode
#define DEBUG false

// Lock base rotation DOF (Add dof in header file if off)
#define LOCKBASE true

// Set Task, if true, then only Gripper is allowed to touch tube, else whole arm
#define GRIPPER_TASK true

namespace gazebo
{
 
// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(ArmPlugin)


// constructor
ArmPlugin::ArmPlugin() : ModelPlugin(), cameraNode(new gazebo::transport::Node()), collisionNode(new gazebo::transport::Node())
{
	printf("ArmPlugin::ArmPlugin()\n");

	for( uint32_t n=0; n < DOF; n++ )
		resetPos[n] = 0.0f;

	resetPos[1] = 0.25;

	for( uint32_t n=0; n < DOF; n++ )
	{
		ref[n] = resetPos[n]; //JOINT_MIN;
		vel[n] = 0.0f;
	}

	agent 	         = NULL;
	inputState       = NULL;
	inputBuffer[0]   = NULL;
	inputBuffer[1]   = NULL;
	inputBufferSize  = 0;
	inputRawWidth    = 0;
	inputRawHeight   = 0;
	actionJointDelta = 0.15f;  //0.15f std - reduced, so that arm doesn't move to fast, 0.10f for 1.task
	actionVelDelta   = 0.1f; //0.1f std
	maxEpisodeLength = 100; // 100
	episodeFrames    = 0;

	newState         = false;
	newReward        = false;
	endEpisode       = false;
	rewardHistory    = 0.0f;
	testAnimation    = true;
	loopAnimation    = false;
	animationStep    = 0;
	lastGoalDistance = 0.0f;
	avgGoalDelta     = 0.0f;
	successfulGrabs  = 0;
	totalRuns        = 0;
}


// Load
void ArmPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) 
{
	printf("ArmPlugin::Load('%s')\n", _parent->GetName().c_str());

	// Store the pointer to the model
	this->model = _parent;
	this->j2_controller = new physics::JointController(model);

	// Create our node for camera communication
	cameraNode->Init();
	
	/*
	/ TODO - Subscribe to camera topic -> DONE
	/
	*/
	
	cameraSub = cameraNode->Subscribe("/gazebo/arm_world/camera/link/camera/image", &ArmPlugin::onCameraMsg, this);

	// Create our node for collision detection
	collisionNode->Init();
		
	/*
	/ TODO - Subscribe to prop collision topic -> DONE
	/
	*/
	
	collisionSub = collisionNode->Subscribe("/gazebo/arm_world/tube/tube_link/my_contact", &ArmPlugin::onCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ArmPlugin::OnUpdate, this, _1));
}


// CreateAgent
bool ArmPlugin::createAgent()
{
	if( agent != NULL )
		return true;

			
	/*
	/ TODO - Create DQN Agent -> DONE
	/
	*/
	
	agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, NUM_ACTIONS, OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY, BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);

	if( !agent )
	{
		printf("ArmPlugin - failed to create DQN agent\n");
		return false;
	}

	// Allocate the python tensor for passing the camera state
		
	inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

	if( !inputState )
	{
		printf("ArmPlugin - failed to allocate %ux%ux%u Tensor\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
		return false;
	}

	return true;
}



// onCameraMsg
void ArmPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
	// don't process the image if the agent hasn't been created yet
	if( !agent )
		return;

	// check the validity of the message contents
	if( !_msg )
	{
		printf("ArmPlugin - recieved NULL message\n");
		return;
	}

	// retrieve image dimensions
	
	const int width  = _msg->image().width();
	const int height = _msg->image().height();
	const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;	// bits per pixel
	const int size   = _msg->image().data().size();

	if( bpp != 24 )
	{
		printf("ArmPlugin - expected 24BPP uchar3 image from camera, got %i\n", bpp);
		return;
	}

	// allocate temp image if necessary
	if( !inputBuffer[0] || size != inputBufferSize )
	{
		if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
		{
			printf("ArmPlugin - cudaAllocMapped() failed to allocate %i bytes\n", size);
			return;
		}

		printf("ArmPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n", width, height, bpp, size);
		
		inputBufferSize = size;
		inputRawWidth   = width;
		inputRawHeight  = height;
	}

	memcpy(inputBuffer[0], _msg->image().data().c_str(), inputBufferSize);
	newState = true;

	if(DEBUG){printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);}

}


// onCollisionMsg
void ArmPlugin::onCollisionMsg(ConstContactsPtr &contacts)
{
	if(DEBUG){printf("collision callback (%u contacts)\n", contacts->contact_size());}

	if( testAnimation )
		return;

	for (unsigned int i = 0; i < contacts->contact_size(); ++i)
	{
      	if( strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0 )
			continue;

		if(DEBUG){std::cout << "Collision between[" << contacts->contact(i).collision1()
			     << "] and [" << contacts->contact(i).collision2() << "]\n";}
      	
		/*
		/ TODO - Check if there is collision between the arm and object, then issue learning reward ->DONE
		/
		*/
		
		bool collisionCheck = false;

// next, used for Task 2: Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy

#if GRIPPER_TASK
		if((strcmp(contacts->contact(i).collision1().c_str(), COLLISION_ITEM) == 0) && (strcmp(contacts->contact(i).collision2().c_str(), COLLISION_POINT) == 0))
		{
          	rewardHistory = REWARD_WIN;
			newReward  = true;
			endEpisode = true;
			std::cout << "Collision with " << contacts->contact(i).collision2() << " \n";
          	return;
        }
      	else
      	{
          	rewardHistory = REWARD_LOSS;
			newReward  = true;
			endEpisode = true;
			std::cout << "Collision with " << contacts->contact(i).collision2() << " \n";
          	return;          
        }
      
  
// next, used for Task 1: Have any part of the robot arm touch the object of interest, with at least a 90% accuracy   

#else
		if( strcmp(contacts->contact(i).collision1().c_str(), COLLISION_ITEM) == 0 )  // collision with any part of the robot arm is allowed
		{
			rewardHistory = REWARD_WIN;
			newReward  = true;
			endEpisode = true;

			return;
		}
#endif	
		
	}
}


// upon recieving a new frame, update the AI agent
bool ArmPlugin::updateAgent()
{
	// convert uchar3 input from camera to planar BGR
	if( CUDA_FAILED(cudaPackedToPlanarBGR((uchar3*)inputBuffer[1], inputRawWidth, inputRawHeight,
							         inputState->gpuPtr, INPUT_WIDTH, INPUT_HEIGHT)) )
	{
		printf("ArmPlugin - failed to convert %zux%zu image to %ux%u planar BGR image\n",
			   inputRawWidth, inputRawHeight, INPUT_WIDTH, INPUT_HEIGHT);

		return false;
	}

	// select the next action
	int action = 0;

	if( !agent->NextAction(inputState, &action) )
	{
		printf("ArmPlugin - failed to generate agent's next action\n");
		return false;
	}

	// make sure the selected action is in-bounds
	if( action < 0 || action >= DOF * 2 )
	{
		printf("ArmPlugin - agent selected invalid action, %i\n", action);
		return false;
	}

	if(DEBUG){printf("ArmPlugin - agent selected action %i\n", action);}



#if VELOCITY_CONTROL
	// if the action is even, increase the joint position by the delta parameter
	// if the action is odd,  decrease the joint position by the delta parameter

		
	/*
	/ TODO - Increase or decrease the joint velocity based on whether the action is even or odd
	/
	*/
	
	float velocity = 0.0; // TODO - Set joint velocity based on whether action is even or odd.

	if( velocity < VELOCITY_MIN )
		velocity = VELOCITY_MIN;

	if( velocity > VELOCITY_MAX )
		velocity = VELOCITY_MAX;

	vel[action/2] = velocity;
	
	for( uint32_t n=0; n < DOF; n++ )
	{
		ref[n] += vel[n];

		if( ref[n] < JOINT_MIN )
		{
			ref[n] = JOINT_MIN;
			vel[n] = 0.0f;
		}
		else if( ref[n] > JOINT_MAX )
		{
			ref[n] = JOINT_MAX;
			vel[n] = 0.0f;
		}
	}
#else
	
	/*
	/ TODO - Increase or decrease the joint position based on whether the action is even or odd ->DONE
	/
	*/
		
  	float joint = ref[action/2]; // TODO - Set joint position based on whether action is even or odd.
  	
  	if (action % 2 == 0)
    {
    	joint = ref[action/2]+actionJointDelta;
      	//std::cout << "angle of joint " << action/2 << " : " << joint << "\n";
    }
  	else
    {
    	joint = ref[action/2]-actionJointDelta; 
    }
  	
	// limit the joint to the specified range
	if( joint < JOINT_MIN )
		joint = JOINT_MIN;
	
	if( joint > JOINT_MAX )
		joint = JOINT_MAX;

	ref[action/2] = joint;

#endif

	return true;
}


// update joint reference positions, returns true if positions have been modified
bool ArmPlugin::updateJoints()
{
	if( testAnimation )	// test sequence
	{
		const float step = (JOINT_MAX - JOINT_MIN) * (float(1.0f) / float(ANIMATION_STEPS));

#if 0
		// range of motion
		if( animationStep < ANIMATION_STEPS )
		{
			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MIN + step * float(animationStep);
		}
		else if( animationStep < ANIMATION_STEPS * 2 )
		{			
			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MAX - step * float(animationStep-ANIMATION_STEPS);
		}
		else
		{
			animationStep = 0;

		}

#else
		// return to base position
		for( uint32_t n=0; n < DOF; n++ )
		{
			
			if( ref[n] < resetPos[n] )
				ref[n] += step;
			else if( ref[n] > resetPos[n] )
				ref[n] -= step;

			if( ref[n] < JOINT_MIN )
				ref[n] = JOINT_MIN;
			else if( ref[n] > JOINT_MAX )
				ref[n] = JOINT_MAX;
			
		}

		animationStep++;
#endif

		// reset and loop the animation
		if( animationStep > ANIMATION_STEPS )
		{
			animationStep = 0;
			
			if( !loopAnimation )
				testAnimation = false;
		}
		else if( animationStep == ANIMATION_STEPS / 2 )
		{	
			ResetPropDynamics();
		}

		return true;
	}

	else if( newState && agent != NULL )
	{
		// update the AI agent when new camera frame is ready
		episodeFrames++;

		if(DEBUG){printf("episode frame = %i\n", episodeFrames);}

		// reset camera ready flag
		newState = false;

		if( updateAgent() )
			return true;
	}

	return false;
}


// get the servo center for a particular degree of freedom
float ArmPlugin::resetPosition( uint32_t dof )
{
	return resetPos[dof];
}


// compute the distance between two bounding boxes
static float BoxDistance(const math::Box& a, const math::Box& b)
{
	float sqrDist = 0;

	if( b.max.x < a.min.x )
	{
		float d = b.max.x - a.min.x;
		sqrDist += d * d;
	}
	else if( b.min.x > a.max.x )
	{
		float d = b.min.x - a.max.x;
		sqrDist += d * d;
	}

	if( b.max.y < a.min.y )
	{
		float d = b.max.y - a.min.y;
		sqrDist += d * d;
	}
	else if( b.min.y > a.max.y )
	{
		float d = b.min.y - a.max.y;
		sqrDist += d * d;
	}

	if( b.max.z < a.min.z )
	{
		float d = b.max.z - a.min.z;
		sqrDist += d * d;
	}
	else if( b.min.z > a.max.z )
	{
		float d = b.min.z - a.max.z;
		sqrDist += d * d;
	}
	
	return sqrtf(sqrDist);
}


// called by the world update start event
void ArmPlugin::OnUpdate(const common::UpdateInfo& updateInfo)
{
	// deferred loading of the agent (this is to prevent Gazebo black/frozen display)
	if( !agent && updateInfo.simTime.Float() > 1.5f )
	{
		if( !createAgent() )
			return;
	}

	// verify that the agent is loaded
	if( !agent )
		return;

	// determine if we have new camera state and need to update the agent
	const bool hadNewState = newState && !testAnimation;

	// update the robot positions with vision/DQN
	if( updateJoints() )
	{
		double angle(1);

#if LOCKBASE
		j2_controller->SetJointPosition(this->model->GetJoint("base"), 	0);
		j2_controller->SetJointPosition(this->model->GetJoint("joint1"),  ref[0]);
		j2_controller->SetJointPosition(this->model->GetJoint("joint2"),  ref[1]);

#else
		j2_controller->SetJointPosition(this->model->GetJoint("base"), 	 ref[0]); 
		j2_controller->SetJointPosition(this->model->GetJoint("joint1"),  ref[1]);
		j2_controller->SetJointPosition(this->model->GetJoint("joint2"),  ref[2]);
#endif
	}

	// episode timeout
	if( maxEpisodeLength > 0 && episodeFrames > maxEpisodeLength )
	{
		printf("ArmPlugin - triggering EOE, episode has exceeded %i frames\n", maxEpisodeLength);
		rewardHistory = REWARD_LOSS;
		newReward     = true;
		endEpisode    = true;
	}

	// if an EOE reward hasn't already been issued, compute an intermediary reward
	if( hadNewState && !newReward )
	{
		PropPlugin* prop = GetPropByName(PROP_NAME);

		if( !prop )
		{
			printf("ArmPlugin - failed to find Prop '%s'\n", PROP_NAME);
			return;
		}

		// get the bounding box for the prop object
		const math::Box& propBBox = prop->model->GetBoundingBox();
		physics::LinkPtr gripper  = model->GetLink(GRIP_NAME);

		if( !gripper )
		{
			printf("ArmPlugin - failed to find Gripper '%s'\n", GRIP_NAME);
			return;
		}

		// get the bounding box for the gripper		
		const math::Box& gripBBox = gripper->GetBoundingBox();
		const float groundContact = 0.00f; //0.05f
		
		/*
		/ TODO - set appropriate Reward for robot hitting the ground. ->DONE
		/
		*/
		
		bool checkGroundContact = false;

		if (gripBBox.min.z <= groundContact || gripBBox.max.z <= groundContact)
		{
			checkGroundContact = true;
		}
		
		if(checkGroundContact)
		{
						
			if(DEBUG){printf("GROUND CONTACT, EOE\n");}
			
			printf("GROUND CONTACT, EOE\n");
			
			rewardHistory = REWARD_LOSS;
			newReward     = true;
			endEpisode    = true;
		}
		
		
		/*
		/ TODO - Issue an interim reward based on the distance to the object ->DONE
		/
		*/ 
		
		
		if(!checkGroundContact)
		{
			const float distGoal = BoxDistance(gripBBox, propBBox); // compute the reward from distance to the goal

			if(DEBUG){printf("distance('%s', '%s') = %f\n", gripper->GetName().c_str(), prop->model->GetName().c_str(), distGoal);}

			
			if( episodeFrames > 1 )
			{
				const float distDelta  = lastGoalDistance - distGoal;
				
				// calculate distances between joints and Objects
              	physics::LinkPtr joint1Base = model->GetLink(JOINT1_NAME);
              	const math::Box& joint1BBox = joint1Base->GetBoundingBox();
              	const float distJoint1Prop = BoxDistance(joint1BBox, propBBox);
              	
              	physics::LinkPtr joint2Base = model->GetLink(JOINT2_NAME);
              	const math::Box& joint2BBox = joint2Base->GetBoundingBox();
              	//const float distJoint2Prop = BoxDistance(joint2BBox, propBBox);
              
              	const float distJoint1Grip = BoxDistance(joint1BBox, gripBBox);
              
              	float rewardDist1 = 0.0f;
              
                //if ((distJoint1Grip) > 1.04 && (distJoint1Grip) < 1.15)
                //{
                	if ((distJoint1Grip) > 1.15f)
                	{
                		rewardDist1 =  1.15f / distJoint1Grip;
                	}
                	else
                	{
                		rewardDist1 = distJoint1Grip / 1.15f;	
                	}
               //}
                //else
                //{
               // 	rewardDist1 = -0.3f;  //-1.0f
               // }

                const float distJoint2GroundMin = joint2BBox.min.z;
                const float distJoint2GroundMax = joint2BBox.max.z;
                float rewardDist2 = 0.0f;
                	
                	if ((distJoint2GroundMin) > 0.42f)
                	{
                		rewardDist2 =  0.42f / distJoint2GroundMin;
                	}
                	else
                	{
                		rewardDist2 =  distJoint2GroundMin / 0.42f - 1.0f;	
                	}
                

                //if ((distJoint2GroundMin) > 0.39 && (distJoint2GroundMin) < 0.49)
                //{

                //	rewardDist2 = 0.3f;   //2.0f

                //}
                //else
                //{
                //	rewardDist2 = -0.2f;  //-0.9f
                //}

                if (distGoal < 0.20)
                {
                	actionJointDelta = 0.15f;
                }
                else
                {
                	actionJointDelta = 0.15f;
                }


                float rewardDist3 = 0.0f;
                if (joint2BBox.min.x < 0 || joint2BBox.max.x < 0)
                {
                	rewardDist3 = -5.0f;
                }

                float rewardDist4 = 0.0f;
                if (gripBBox.max.x < 0 || gripBBox.min.x < 0)
                {
                	rewardDist4 = -5.0f;
                }

                  
// next, used for Task 2: Have only the gripper base of the robot arm touch the object, with at least a 80% accuracy

#if GRIPPER_TASK
				float EnergyLvl = ((MAX_DIST_2 - distGoal) * (MAX_DIST_2 - distGoal)) / (MAX_DIST_2 * MAX_DIST_2) * distDelta;
              	//avgGoalDelta  = avgGoalDelta + EnergyLvl;
				//rewardHistory = EnergyLvl * REWARD_WIN;
              	avgGoalDelta  = (avgGoalDelta * ALPHA) + (distDelta * (1.0f - ALPHA)); //ALPHA
              	//rewardHistory = exp(-GAMMA_FALLOFF * distGoal) * 0.1f; 
              	//rewardHistory = 0 * avgGoalDelta * REWARD_INTERIM + rewardDist1; 
              	//if (avgGoalDelta>0)
              	//{
              	//	rewardDist1 = 1.0f;
              	//}
              	
              	if (avgGoalDelta > 0)
              	{
              		rewardHistory = avgGoalDelta*REWARD_INTERIM;
              		//rewardHistory = REWARD_INTERIM;
              		std::cout << "rewardHistory avgGoalDelta > 0 " << rewardHistory << "\n";
              	}
              	else
              	{
              		rewardHistory = REWARD_INTERIM*avgGoalDelta;
              		//ewardHistory = -REWARD_INTERIM;
              		std::cout << "rewardHistory avgGoalDelta < 0 " << rewardHistory << "\n";
              	}
              	
              	//rewardHistory = avgGoalDelta * REWARD_INTERIM; // * rewardDist1;
              	
              	if (abs(distDelta*100) <= 0)
              	{
              		//std::cout << "true " << abs(distDelta) << "\n";	
              		rewardHistory = rewardHistory - 0.5f * REWARD_INTERIM;
              		std::cout << "Keine Bewegung, Faktor der abgezogen wird " << 1.1f * REWARD_INTERIM << "\n";
              	}

              	else
              	{
              	rewardHistory = (MAX_DIST_2 - distGoal) / MAX_DIST_2 * rewardHistory * rewardDist1 * rewardDist2; // + rewardDist4;               		
              	}

              	std::cout << "rewardHistory vor Rechnung " << rewardHistory << "\n";

              	//rewardHistory = (MAX_DIST_2 - distGoal) / MAX_DIST_2 * rewardHistory * rewardDist1 * rewardDist2; // + rewardDist4; 

              	newReward     = true;
              	//std::cout << "lastGoalDistance " << lastGoalDistance << "\n";
              	//std::cout << "distGoal " << distGoal << "\n";
              	//std::cout << "distDelta " << distDelta << "\n";
              	//std::cout << "avgGoalDelta " << avgGoalDelta << "\n";
                //std::cout << "rewardDist1 " << rewardDist1 << "\n";
                //std::cout << "EnergyLvl " << EnergyLvl << "\n";
              	
              	std::cout << "rewardHistory Gesamt " << rewardHistory << "\n";
              	//std::cout << "rewardDist4 " << rewardDist4 << "\n";
              	std::cout << "je naeher am Ziel, desto mehr Punkte " << 0.7f*pow(((MAX_DIST_2 - distGoal) / MAX_DIST_2),4) << "\n";
              	std::cout << "rewardDist1 Abstand Griper zu Joint" << rewardDist1 << "\n";
              	std::cout << "rewardDist2 Abstand joint 2 zum Boden " << rewardDist2 << "\n";
              	std::cout << "ref0 " << ref[0] << "\n";
              	std::cout << "ref1 " << ref[1] << "\n";
              	std::cout << "ref2 " << ref[2] << "\n";
              	//std::cout << "Distance betw joint 1 and Object " << distJoint1Prop << "\n";
                //std::cout << "Distance betw joint 2 and ob " << distJoint2Prop << "\n";
                //std::cout << "Distance betw joint 1 and Grip " << distJoint1Grip << "\n";
                //std::cout << "Distance betw joint 2 and Ground min " << distJoint2GroundMin << "\n";
				//std::cout << "Distance betw joint 2 and Ground max " << distJoint2GroundMax << "\n";                

// next, used for Task 1: Have any part of the robot arm touch the object of interest, with at least a 90% accuracy   
#else
				//float EnergyLvl = ((MAX_DIST_1 - distGoal) * (MAX_DIST_1 - distGoal)) / (MAX_DIST_1 * MAX_DIST_1) * ALPHA;
 				// compute the smoothed moving average of the delta of the distance to the goal             	
              	avgGoalDelta  = (avgGoalDelta * ALPHA) + (distDelta * (1.0f - ALPHA));	
				//rewardHistory = avgGoalDelta * REWARD_WIN;             	
              	rewardHistory = avgGoalDelta * REWARD_INTERIM; 
              	newReward     = true;
#endif

			}

			lastGoalDistance = distGoal;
		} 
	}

	// issue rewards and train DQN
	if( newReward && agent != NULL )
	{
		if(DEBUG){printf("ArmPlugin - issuing reward %f, EOE=%s  %s\n", rewardHistory, endEpisode ? "true" : "false", (rewardHistory > 0.1f) ? "POS+" :(rewardHistory > 0.0f) ? "POS" : (rewardHistory < 0.0f) ? "    NEG" : "       ZERO");}
		agent->NextReward(rewardHistory, endEpisode);

		// reset reward indicator
		newReward = false;

		// reset for next episode
		if( endEpisode )
		{
			testAnimation    = true;	// reset the robot to base position
			loopAnimation    = false;
			endEpisode       = false;
			episodeFrames    = 0;
			lastGoalDistance = 0.0f;
			avgGoalDelta     = 0.0f;
            //std::cout << "Reset true \n";

			// track the number of wins and agent accuracy
			if( rewardHistory >= REWARD_WIN )
				successfulGrabs++;

			totalRuns++;
			printf("Current Accuracy:  %0.4f (%03u of %03u)  (reward=%+0.2f %s)\n", float(successfulGrabs)/float(totalRuns), successfulGrabs, totalRuns, rewardHistory, (rewardHistory >= REWARD_WIN ? "WIN" : "LOSS"));


			for( uint32_t n=0; n < DOF; n++ )
				vel[n] = 0.0f;
		}
	}
}

}

