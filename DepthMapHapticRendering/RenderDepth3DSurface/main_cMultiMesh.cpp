//==============================================================================
/*
    \author    Yitian Shao
	\templete from <http://www.chai3d.org>
	\created 10/13/2015 

	This is the second step on realizing depth map haptic rendering.
	The goal here is to import a depth map image in format like '.png' and 
	construct a 3D surface through a mesh. Then rendering it.
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "chai3d.h"
//#include "png.h" // use libpng to load png image?
//#include <cstdio> // use libpng to load png image?
//------------------------------------------------------------------------------
using namespace chai3d;
using namespace std;
//------------------------------------------------------------------------------
#ifndef MACOSX
#include "GL/glut.h"
#else
#include "GLUT/glut.h"
#endif
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// GENERAL SETTINGS
//------------------------------------------------------------------------------

// stereo Mode
/*
    C_STEREO_DISABLED:            Stereo is disabled 
    C_STEREO_ACTIVE:              Active stereo for OpenGL NVDIA QUADRO cards
    C_STEREO_PASSIVE_LEFT_RIGHT:  Passive stereo where L/R images are rendered next to each other
    C_STEREO_PASSIVE_TOP_BOTTOM:  Passive stereo where L/R images are rendered above each other
*/
cStereoMode stereoMode = C_STEREO_DISABLED;

// fullscreen mode
bool fullscreen = false;

// check whether file loaded or not
bool fileload;

//------------------------------------------------------------------------------
// DECLARED VARIABLES
//------------------------------------------------------------------------------

// a world that contains all objects of the virtual environment
cWorld* world;

// a camera to render the world in the window display
cCamera* camera;

// a light source to illuminate the objects in the world
cDirectionalLight* light;

// a haptic device handler
cHapticDeviceHandler* handler;

// a pointer to the current haptic device
cGenericHapticDevicePtr hapticDevice;

// a label to display the position [m] of the haptic device
cLabel* labelHapticDevicePosition;

// a global variable to store the position [m] of the haptic device
cVector3d hapticDevicePosition;

// a global variable to store the velocity [m/s] of the haptic device
cVector3d hapticDeviceVelocity;

// a label to display the rate [Hz] at which the simulation is running
cLabel* labelHapticRate;

// a virtual tool representing the haptic device in the scene
cToolCursor* tool;

/*10/13/2015-Add 3D object to display depth map as displacement map*/
cMultiMesh* object;

/*11/13/2015 a mesh object for debbuging*/
cMesh* plane0;

/*11/13/2015 a depth image*/
cImage* depthImage;

/*11/13/2015 a label for debugging purpose*/
cLabel* debuggingLabel;

/*11/13/2015 a depth value*/
cColorb depthValue;

// flag to indicate if the haptic simulation currently running
bool simulationRunning = false;

// flag to indicate if the haptic simulation has terminated
bool simulationFinished = false;

// frequency counter to measure the simulation haptic rate
cFrequencyCounter frequencyCounter;

// information about computer screen and GLUT display window
int screenW;
int screenH;
int windowW;
int windowH;
int windowPosX;
int windowPosY;

/*11/13/2015 an matrix containing depth values*/
float depthMat[540][960];

//------------------------------------------------------------------------------
// DECLARED FUNCTIONS
//------------------------------------------------------------------------------

// callback when the window display is resized
void resizeWindow(int w, int h);

// callback when a key is pressed
void keySelect(unsigned char key, int x, int y);

// callback to render graphic scene
void updateGraphics(void);

// callback of GLUT timer
void graphicsTimer(int data);

// function that closes the application
void close(void);

// main haptics simulation loop
void updateHaptics(void);

int main(int argc, char* argv[])
{
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    cout << endl;
    cout << "-----------------------------------" << endl;
    cout << "CHAI3D" << endl;
    cout << "-----------------------------------" << endl << endl << endl;
    cout << "Keyboard Options:" << endl << endl;
    cout << "[f] - Enable/Disable full screen mode" << endl;
    cout << "[x] - Exit application" << endl;
    cout << endl << endl;

    //--------------------------------------------------------------------------
    // OPENGL - WINDOW DISPLAY
    //--------------------------------------------------------------------------

    // initialize GLUT
    glutInit(&argc, argv);

    // retrieve  resolution of computer display and position window accordingly
    screenW = glutGet(GLUT_SCREEN_WIDTH);
    screenH = glutGet(GLUT_SCREEN_HEIGHT);
    windowW = (int)(0.8 * screenH);
    windowH = (int)(0.5 * screenH);
    windowPosY = (screenH - windowH) / 2;
    windowPosX = windowPosY; 

    // initialize the OpenGL GLUT window
    glutInitWindowPosition(windowPosX, windowPosY);
    glutInitWindowSize(windowW, windowH);
    if (stereoMode == C_STEREO_ACTIVE)
    {
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE | GLUT_STEREO);
    }
    else
    {
        glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
    }

    // create display context and initialize GLEW library
    glutCreateWindow(argv[0]);
    glewInit();

    // setup GLUT options
    glutDisplayFunc(updateGraphics);
    glutKeyboardFunc(keySelect);
    glutReshapeFunc(resizeWindow);
    glutSetWindowTitle("Render Depth Map 3D Surface");
	
	//================just for test ==========================
	//const char * imagePath = "../bin/resources/images/rabbit.tif";

    // set fullscreen mode
    if (fullscreen)
    {
        glutFullScreen();
    }

	//--------------------------------------------------------------------------
	// LOAD FILE [11/13/2015 - present]
	//--------------------------------------------------------------------------
	
	//=====================================================================================================
	// create a new image
	depthImage = new cImage(windowW, windowH, GL_RGB, GL_UNSIGNED_INT);

	// import the depth image from a .png file
	fileload = depthImage->loadFromFile("../bin/resources/image/rabbit.png");
	if (!fileload)
	{
		cout << "Error - Image failed to load correctly." << endl;
		close();
		return (-1);
	}

	//--------------------------------------------------------------------------
	// Read depth value from each pixel of the depth image [11/13/2015 - present]
	//--------------------------------------------------------------------------
	// check whether the command is succefully executed
	bool isRight;
	//int dpVal[4];

	// depth map row (Height) and column (Width) length
	int mapRowNum, mapColNum;
	mapRowNum = depthImage->getHeight();
	mapColNum = depthImage->getWidth();

	for (int i = 0; i < mapRowNum; i++)
	{
		for (int j = 0; j < mapColNum; j++)
		{
			isRight = depthImage->getPixelColor(j, i, depthValue);

			if (isRight == false)
			{
				cout << "error - failed! [" + to_string(i) + "," + to_string(j) + "]" << endl;
			}

			// For gray scale image, R = G = B = GrayScale
			depthMat[i][j] = depthValue.m_color[0]; // Extract R Value as depth information

			depthMat[i][j] = depthMat[i][j] / 255;	// Normalize the depth image to [0 , 1] scale

			//if (depthMat[i][j] < 0.2) // Used for checking the depth values
			//{
			//	cout << "Position[" + to_string(i) + "," + to_string(j) + "] = " + to_string(depthMat[i][j]) << endl;
			//}
		}
		cout << endl;
	}

	//======================================================================================================

    //--------------------------------------------------------------------------
    // WORLD - CAMERA - LIGHTING
    //--------------------------------------------------------------------------

	// create a new world.
	world = new cWorld();

	// set the background color of the environment
	world->m_backgroundColor.setBlack();

	// create a camera and insert it into the virtual world
	camera = new cCamera(world);
	world->addChild(camera);

	// position and orient the camera
	camera->set(cVector3d(0.6, 0.0, 0),    // camera position (eye)
		cVector3d(0.0, 0.0, 0.0),    // lookat position (target)
		cVector3d(0.0, 0.0, 1.0));   // direction of the (up) vector

	// set the near and far clipping planes of the camera
	// anything in front/behind these clipping planes will not be rendered
	camera->setClippingPlanes(0.01, 100);

	// set stereo mode
	camera->setStereoMode(stereoMode);

	// set stereo eye separation and focal length (applies only if stereo is enabled)
	camera->setStereoEyeSeparation(0.03);
	camera->setStereoFocalLength(1.5);

	// enable multi-pass rendering to handle transparent objects
	camera->setUseMultipassTransparency(true);

	// create a light source
	light = new cDirectionalLight(world);

	// attach light to camera
	camera->addChild(light);

	// enable light source
	light->setEnabled(true);

	// define the direction of the light beam
	light->setDir(-3.0, -0.5, 0.0);

	// set lighting conditions
	light->m_ambient.set(0.1, 0.1, 0.1);
	light->m_diffuse.set(0.4, 0.4, 0.4);
	light->m_specular.set(1.0, 1.0, 1.0);

    //--------------------------------------------------------------------------
    // HAPTIC DEVICE
    //--------------------------------------------------------------------------

    // create a haptic device handler
    handler = new cHapticDeviceHandler();

    // get a handle to the first haptic device
    handler->getDevice(hapticDevice, 0);

    // open a connection to haptic device
    hapticDevice->open();

    // calibrate device (if necessary)
    hapticDevice->calibrate();

    // retrieve information about the current haptic device
    cHapticDeviceInfo info = hapticDevice->getSpecifications();

	// create a tool (cursor) and insert into the world
	tool = new cToolCursor(world);
	world->addChild(tool);

	// connect the haptic device to the virtual tool
	tool->setHapticDevice(hapticDevice);

	// define the radius of the tool (sphere)
	double toolRadius = 0.005;

	// define a radius for the tool
	tool->setRadius(toolRadius);

	// hide the device sphere. only show proxy.
	tool->setShowContactPoints(true, false);

	// create a white cursor
	tool->m_hapticPoint->m_sphereProxy->m_material->setRed();

	// map the physical workspace of the haptic device to a larger virtual workspace.
	tool->setWorkspaceRadius(1.0);

	// start the haptic tool
	tool->start();

	//--------------------------------------------------------------------------
	// CREATE OBJECT
	// Import a Depth image instead of a 3D displacement map
	//--------------------------------------------------------------------------

	// read the scale factor between the physical workspace of the haptic
	// device and the virtual workspace defined for the tool
	double workspaceScaleFactor = tool->getWorkspaceScaleFactor();

	// stiffness properties
	double maxStiffness = info.m_maxLinearStiffness / workspaceScaleFactor;

	// create a virtual mesh
	object = new cMultiMesh();

	// add object to world
	world->addChild(object);

	/////////////////////////////////////////////////////////////////////////
	// PLANE 0:
	/////////////////////////////////////////////////////////////////////////
	const int PlaneW = 1.6;
	const int PlaneH = 0.9;

	// create a mesh
	plane0 = new cMesh();

	// create plane
	cCreatePlane(plane0, 0.3, 0.3);

	// create collision detector
	plane0->createAABBCollisionDetector(toolRadius);

	// add object to world
	world->addChild(plane0);

	// set the position of the object
	plane0->setLocalPos(-0.8, -0.45, 0.0);

	// set graphic properties
	//plane0->m_texture = cTexture2d::create();
	//plane0->m_texture = ;

	cColorf color0;
	color0.setBlueCadet();

	for (int i = 0; i < mapRowNum; i++)
	{
		for (int j = 0; j < mapColNum; j++)
		{
			unsigned int vertexIndex0 = plane0->newVertex((-0.8 + 0.0016*j), (-0.45 + 0.0016*i), depthMat[i][j], 1.0, 0.0, 0.0, (-0.8 + 0.0016*j), (-0.45 + 0.0016*i), depthMat[i][j]);

			// define vertex color			
			plane0->m_vertices->setColor(vertexIndex0, color0);
	
		}
	}

	//Original method in 10/13/2015 : Import a 3D displacement map (object)
	//fileload = object->loadFromFile("../bin/resources/model/displacement_map.obj"); 
	//(x-up,z-forward output)

	// disable culling so that faces are rendered on both sides
	//object->setUseCulling(false);

	// get dimensions of object
	object->computeBoundaryBox(true);
	double size = cSub(object->getBoundaryMax(), object->getBoundaryMin()).length();

	// resize object to screen
	if (size > 0.001)
	{
		object->scale(1.0 / size);
	}

	// compute a boundary box
	object->computeBoundaryBox(true);

	// show/hide bounding box
	object->setShowBoundaryBox(false);

	// compute collision detection algorithm
	object->createAABBCollisionDetector(toolRadius);

	// define a default stiffness for the object
	object->setStiffness(0.2 * maxStiffness, true);

	// define some haptic friction properties
	object->setFriction(0.1, 0.2, true);

	// enable display list for faster graphic rendering
	object->setUseDisplayList(true);

	// center object in scene
	object->setLocalPos(-1.0 * object->getBoundaryCenter());

    //--------------------------------------------------------------------------
    // WIDGETS
    //--------------------------------------------------------------------------

    // create a font
    cFont *font = NEW_CFONTCALIBRI20();
    
	// create a label to display the position of haptic device
	labelHapticDevicePosition = new cLabel(font);
	camera->m_frontLayer->addChild(labelHapticDevicePosition);

    // create a label to display the haptic rate of the simulation
    labelHapticRate = new cLabel(font);
    camera->m_frontLayer->addChild(labelHapticRate);

	//=============================================================================
	// add a real-time debugging label
	//debuggingLabel = new cLabel(font);
	//camera->m_frontLayer->addChild(debuggingLabel);
	//=============================================================================

    //--------------------------------------------------------------------------
    // START SIMULATION
    //--------------------------------------------------------------------------

    // create a thread which starts the main haptics rendering loop
    cThread* hapticsThread = new cThread();
    hapticsThread->start(updateHaptics, CTHREAD_PRIORITY_HAPTICS);

    // start the main graphics rendering loop
    glutTimerFunc(50, graphicsTimer, 0);
    glutMainLoop();

    // close everything
    close();

    // exit
    return (0);
}

//------------------------------------------------------------------------------

void resizeWindow(int w, int h)
{
    windowW = w;
    windowH = h;
}

//------------------------------------------------------------------------------

void keySelect(unsigned char key, int x, int y)
{
    // option ESC: exit
    if ((key == 27) || (key == 'x'))
    {
        close();
        exit(0);
    }

    // option f: toggle fullscreen
    if (key == 'f')
    {
        if (fullscreen)
        {
            windowPosX = glutGet(GLUT_INIT_WINDOW_X);
            windowPosY = glutGet(GLUT_INIT_WINDOW_Y);
            windowW = glutGet(GLUT_INIT_WINDOW_WIDTH);
            windowH = glutGet(GLUT_INIT_WINDOW_HEIGHT);
            glutPositionWindow(windowPosX, windowPosY);
            glutReshapeWindow(windowW, windowH);
            fullscreen = false;
        }
        else
        {
            glutFullScreen();
            fullscreen = true;
        }
    }
}

//------------------------------------------------------------------------------

void close(void)
{
    // stop the simulation
    simulationRunning = false;

    // wait for graphics and haptics loops to terminate
    while (!simulationFinished) { cSleepMs(100); }

    // close haptic device
    hapticDevice->close();
}

//------------------------------------------------------------------------------

void graphicsTimer(int data)
{
    if (simulationRunning)
    {
        glutPostRedisplay();
    }

    glutTimerFunc(50, graphicsTimer, 0);
}

//------------------------------------------------------------------------------

void updateGraphics(void)
{
    /////////////////////////////////////////////////////////////////////
    // UPDATE WIDGETS
    /////////////////////////////////////////////////////////////////////

	// display new position data
	labelHapticDevicePosition->setString("position [m]: " + hapticDevicePosition.str(3));

    // display haptic rate data
    labelHapticRate->setString ("haptic rate: "+cStr(frequencyCounter.getFrequency(), 0) + " [Hz]");

    // update position of label
    labelHapticRate->setLocalPos((int)(0.5 * (windowW - labelHapticRate->getWidth())), 15);

	//======================================================================
	// display real-time debugging label 
	//debuggingLabel->setString("Image Pixel Value = " + to_string());
	//debuggingLabel->setLocalPos((int)(0.5 * (windowW - debuggingLabel->getWidth())), 40);

	//======================================================================


    /////////////////////////////////////////////////////////////////////
    // RENDER SCENE
    /////////////////////////////////////////////////////////////////////

    // render world
    camera->renderView(windowW, windowH);

    // swap buffers
    glutSwapBuffers();

    // check for any OpenGL errors
    GLenum err;
    err = glGetError();
    if (err != GL_NO_ERROR) cout << "Error:  %s\n" << gluErrorString(err);
}

//------------------------------------------------------------------------------

void updateHaptics(void)
{
    // initialize frequency counter
    frequencyCounter.reset();

    // simulation in now running
    simulationRunning  = true;
    simulationFinished = false;

    // main haptic simulation loop
    while(simulationRunning)
    {
        /////////////////////////////////////////////////////////////////////
        // READ HAPTIC DEVICE
        /////////////////////////////////////////////////////////////////////

        // read position 
        cVector3d position;
        hapticDevice->getPosition(position);

        // read user-switch status (button 0)
        bool button = false;
        hapticDevice->getUserSwitch(0, button);


        /////////////////////////////////////////////////////////////////////
        // HAPTIC RENDERING
        /////////////////////////////////////////////////////////////////////

		// compute global reference frames for each object
		world->computeGlobalPositions(true);

		// update position and orientation of tool
		tool->updatePose();

		// compute interaction forces
		tool->computeInteractionForces();

		/////////////////////////////////////////////////////////////////////
		// UPDATE widgets
		/////////////////////////////////////////////////////////////////////

		// update global variable for graphic display update
		hapticDevicePosition = position;

        /////////////////////////////////////////////////////////////////////
        // APPLY FORCES
        /////////////////////////////////////////////////////////////////////

        // send computed force to haptic device
		tool->applyForces();

        // update frequency counter
        frequencyCounter.signal(1);
    }
    
    // exit haptics thread
    simulationFinished = true;
}

//------------------------------------------------------------------------------
