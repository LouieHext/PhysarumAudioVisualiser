#pragma once

#include "ofMain.h"
#include "ofxGui.h"
#include "maxiMFCC.h"
#include "ofxMaxim.h"

//globals for simulation and image size
#define W 1080 
#define H 1080

class ofApp : public ofBaseApp{

	public:

		//main functions
		void setup();
		void update();
		void draw();

		void setupShaders();
		void setupParams();

		void updateAgents();
		void updatePheromones();

		void showFPS();
		void saveFrame();
		//interations
		void keyPressed(int key);
		
		
		//shader objects
		ofShader simulation,diffusion;
		
		//particle struct
		struct Particle {
			glm::vec3 pos;
			float heading;
		};

		//container for particles
		vector<Particle> particles;

		//buffer objects and textures
		ofBufferObject particlesBuffer, particlesBufferClear, pheremones, pheremonesBack,pheremonesClear;
		ofTexture pheromoneIntensityTexture;

		//CPU side array for buffer decloration
		float pheremonesCPU[W*H];

		//GUI and params
		ofxPanel gui;
		ofParameter<float> maxSpeed,turningSpeed;
		ofParameter<float> sensorAngle;
		ofParameter<int> sensorDistance, sensorSize;
		ofParameter<int> numAgents;
		ofParameter<float> decayWeight, diffusionWeight;
		ofParameter<int> densitySpeed,colouring;
		ofParameter<float> baseMulti, densityMulti;

		ofParameterGroup agentSettings, pheromoneSettings;

		//saving
		bool saving;
		ofPixels pixels;
		ofImage image;
		int c = 1;




		//setup helpers
		void maxSetup();
		


		//calculators
		struct Result {
			float timeShortAverage; int minBin; int maxBin; float binFreq; float pitchCentroid;
		};

		Result getAudioFeatures();
		float fftLongAvg(float timeAverageShort);
		float maxPitchBinLongAverage(float maxPitchBin);
		bool beatDetctor(float timeAverageShort, float timeAverageLong);



		//visualisers
		void drawFFT(int minBin, int maxBin, float binFreq, bool onBeat);
		void drawOct();
		void drawInfo(float timeAverageShort);

		/* audio stuff */
		void audioOut(ofSoundBuffer& output) override; //output method
		void audioIn(ofSoundBuffer& input) override; //input method
		ofSoundStream soundStream;
		float* lAudioIn; /* inputs */
		float* rAudioIn;
		int		sampleRate;


		//MAXIMILIAN STUFF:
		double wave, sample, outputs[2], ifftVal;
		maxiMix mymix;
		maxiOsc osc;
		ofxMaxiFFTOctaveAnalyzer oct;

		int nAverages;
		float* ifftOutput;
		int ifftSize;

		float peakFreq = 0;
		float centroid = 0;
		float RMS = 0;

		double averageValue, pitchValue;

		vector<float> timeAverage;
		vector<float> timeAveragePitch;
		vector<float> timeAverageRMS;

		ofxMaxiFFT mfft;
		int fftSize;
		int bins, dataSize;


		//GUI STUFF
		bool bHide;
		int beat;


		ofxPanel guiShader;
		ofxFloatSlider scale, timeScale, powerFBM, retro;
		ofxIntSlider numFBM;

		ofxPanel guiBeat;
		ofxFloatSlider minFreq, maxFreq, detectThreshold, DetectMultiplier;
		ofxIntSlider timeAverageLength, pitchAverageLength;
		ofxFloatSlider pitchMax, rmsMax;


		//visuals

		ofShader shader;
		ofPixels pixels;
		ofFbo canvasFbo;

		int x, y, offset;

		ofTrueTypeFont myfont, myfontBig;

		float horizWidth;
		float horizOffset;
		float vertOffset;
		int debugWidth, debugHeight;
		int maxPitch;


};
