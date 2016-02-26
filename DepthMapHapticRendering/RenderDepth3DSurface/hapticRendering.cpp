//==============================================================================
/*
    \author    Yitian Shao
	\templete from <http://www.chai3d.org>
	\created 10/13/2015 v1.00
	This is the second step on realizing depth map haptic rendering.
	The goal here is to import a depth map image in format like '.png' and 
	construct a 3D surface through a mesh. Then rendering it.

	\updated 11/24/2015 v1.01
	Include depth image processing algorithm in separate file. This file only 
	includes 3D graphics and haptics rendering pipelines.

	\updated 02/14/2016 v1.02
	This is the third step on realizing depth map haptic rendering.
	Bas-relief using gradient compression and Multigrid method succeed.
	Image processing and haptic rendering programs are separated.

	As of 02/14/2016, no more change will be applied to this file.
	Unless 3D and haptic rendering mechanism need to be modified.
*/
//==============================================================================

//------------------------------------------------------------------------------
#include "hapticRendering.h"

#include <iostream> // For test only
#include <fstream> // For test only
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

cMesh* object; // Mesh object 11/19/2015

/*11/13/2015 a label for debugging purpose*/
cLabel* debuggingLabel;

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

// find the depth range of a depth image
double getDepthRange(MMatrix* depthMat);

void hapticRender(MMatrix& renderMat, int argc, char* argv[])
{
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    cout << endl;
    cout << "-----------------------------------" << endl;
    cout << "CHAI3D based Depth Map Haptics Rendering Project" << endl;
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

    // set fullscreen mode
    if (fullscreen)
    {
        glutFullScreen();
    }

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
	// Camera shift additionally by 1.0 on z direction
	camera->set(cVector3d(1.0 + 1.4, 0.0, 0.0),    // camera position (eye) 
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

	// disable vertical mirrored display mode
	camera->setMirrorVertical(false);

	// create a light source
	light = new cDirectionalLight(world);

	// attach light to camera
	camera->addChild(light);

	// enable light source
	light->setEnabled(true);

	// define the direction of the light beam
	light->setDir(-2.0, 0.0, 1.0); 

	// set lighting conditions
	light->m_ambient.set(0.5, 0.5, 0.5); // (0.1) Increase the ambient of light (02/15/2016)
	light->m_diffuse.set(0.8, 0.8, 0.8); // (0.5) Increase the diffuse of light (02/15/2016)
	light->m_specular.set(0.05, 0.05, 0.05); // (1.0) Decrese the specular of light (02/15/2016)

	/* Display detailed parameters of generic objects (for test only) */
	/*cVector3d camPosi = camera->getGlobalPos();
	cout << "camera global potion = " << camPosi.str(3) << endl;*/

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

	// define the radius of the collision dectection ==============================
	double hapticRadius = 0.005;

	// define the radius of the tool (displayed sphere)	===========================
	double toolRadius = 0.01; // (0.005) Displayed too radius doubled 02/26/2016

	// define the stiffness of the contact object =================================
	double stiffnessScale = 0.5; // modified on 11/24/2015

	// define a radius for the tool
	tool->setRadius(toolRadius);

	// show the (device sphere, proxy).
	tool->setShowContactPoints(true, true);

	// create a red cursor
	tool->m_hapticPoint->m_sphereProxy->m_material->setRed();

	// defines the size of the virtual workspace covered by the haptic device
	tool->setWorkspaceRadius(1.6); // It will be changed after constructed the mesh object?

	// start the haptic tool
	tool->start();

	/////////////////////////////////////////////////////////////////////////
	// Mesh object:
	/////////////////////////////////////////////////////////////////////////

	// read the scale factor between the physical workspace of the haptic
	// device and the virtual workspace defined for the tool
	double workspaceScaleFactor = tool->getWorkspaceScaleFactor();

	// stiffness properties
	double maxStiffness = info.m_maxLinearStiffness / workspaceScaleFactor;

	// create a virtual mesh
    object = new cMesh();

    // add object to world
    world->addChild(object);

    // set the position of the object at the center of the world
    object->setLocalPos(1.0, 0.0, 0.0); // Object shift by 1.4 on z direction (1.4) changed 01/15/2016

    // Since we want to see our polygons from both sides, we disable culling.
    object->setUseCulling(false);

    // set color properties
    object->m_material->setBlueCornflower();

    // set stiffness
	object->m_material->setStiffness(stiffnessScale*maxStiffness);

    // enable haptic shading
    object->m_material->setUseHapticShading(true);

    // use display list to increase graphical rendering performance
    object->setUseDisplayList(true);

	//--------------------------------------------------------------------------
	// Construct the mesh object

	// get the size of the image
	/*int sizeX = imageSize[1];
	int sizeY = imageSize[0];*/
	int sizeX = renderMat.getColsNum();
	int sizeY = renderMat.getRowsNum();

	// get the depth range of the image
	double sizeZ = getDepthRange(&renderMat);

	// we look for the largest side
	int largestSide = cMax(sizeX, sizeY);

	// scale the image to fit the world
	double scaleXY = 1.0 / (double)largestSide;
	double scaleZ = 0.2 / sizeZ; // (1.0) changed 01/15/2016

	// we will create an triangle based object. For centering puposes we
	// compute an offset for axis X and Y corresponding to the half size
	// of the image map.
	double offsetX = 0.5 * (double)sizeX * scaleXY;
	double offsetY = 0.5 * (double)sizeY * scaleXY;

	// allocate vertices for this map
	object->m_vertices->newVertices(sizeX*sizeY);

	// set position of each vertex
	int x, y, index;
	index = 0;
	for (y = 0; y<sizeY; y++)
	{
		for (x = 0; x<sizeX; x++)
		{
			// compute the position of the vertex
			double px = scaleXY * (double)x - offsetX;
			double py = scaleXY * (double)y - offsetY;
			//double pz = -scaleZ * mappedMatrix[y][x];
			double pz = -scaleZ * renderMat.getElement(y, x);

			// set vertex position
			object->m_vertices->setLocalPos(index, pz, px, py);
			index++;
		}
	}	

	// Create a triangle based map using the above pixels
	for (x = 0; x<(sizeX - 1); x++)
	{
		for (y = 0; y<(sizeY - 1); y++)
		{
			// get the indexing numbers of the next four vertices
			unsigned int index00 = ((y + 0) * sizeX) + (x + 0);
			unsigned int index01 = ((y + 0) * sizeX) + (x + 1);
			unsigned int index10 = ((y + 1) * sizeX) + (x + 0);
			unsigned int index11 = ((y + 1) * sizeX) + (x + 1);

			// create two new triangles
			object->newTriangle(index00, index01, index10);
			object->newTriangle(index10, index01, index11);
		}
	}

	// compute normals
	object->computeAllNormals();

	// compute bounding box
	object->computeBoundaryBox(true);
	cVector3d min = object->getBoundaryMin();
	cVector3d max = object->getBoundaryMax();

	// compute size of object (largest side)
	cVector3d span = cSub(max, min);
	double size = cMax(span.x(), cMax(span.y(), span.z()));

	// scale object
	const double DESIRED_MESH_SIZE = 2.0;
	double scaleFactor = DESIRED_MESH_SIZE / size;
	object->scale(scaleFactor);

	// compute boundary box again
	object->computeBoundaryBox(true);

	// create collision detector for haptics interaction
	object->createAABBCollisionDetector(1.01 * hapticRadius);

	//--------------------------------------------------------------------------

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
	//labelHapticDevicePosition->setString("position [m]: " + hapticDevicePosition.str(3)); // ?
	labelHapticDevicePosition->setString("position [m]: " + tool->getDeviceGlobalPos().str(3));

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
        //cVector3d position;
        //hapticDevice->getPosition(position); // get position ?

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
		//hapticDevicePosition = position;

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

double getDepthRange(MMatrix* depthMat)
{
	double minVal = 0.0;
	double maxVal = 0.0;

	size_t height = depthMat->getRowsNum();
	size_t width = depthMat->getColsNum();

	for (uint i = 0; i < height; i++)
	{
		for (uint j = 0; j < width; j++)
		{
			if (depthMat->getElement(i, j) < minVal) minVal = depthMat->getElement(i, j);
			if (depthMat->getElement(i, j) > maxVal) maxVal = depthMat->getElement(i, j);
		}
	}
	//cout << "min = " << minVal << " , max = " << maxVal << endl; // for test only

	return (maxVal - minVal);
}