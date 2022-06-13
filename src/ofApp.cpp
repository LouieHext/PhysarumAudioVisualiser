#include "ofApp.h"

//Created 04/2022 - Louie Hext
//Explorations of stigmergy via a Physarum Polycephalum slime mould simulation
//this code  uses a compute shader to simulate the stigmergy properties of physarum slimes
//effectivly motion driven by pheremones

//this code simulates a large number of agents that move in a periodically bound system
//as an agent moves it leaves a pheromone trail behind it.
//imporantly an agents heading is dictated by pheomone intensities in its vicinity.
//this feedback loop causes the agents to follow each other in twisting arcing paths.

//the patterns that these paths make are influenced by the agents global settings.
//exploring these gives a wealth of different structures, some are "physical" in the sense that
//the paths are connected and naturally organic
//other parts of the paramter space act like rain drops, particles, worms, lattices

//for slime like settings try
//speed < distance , turning speed <1.5, angle <1.5

//if speed is > distance you will get streaking particles that wiggle around
//adjust other settings in this part of the param space to explore these

//if turning speed >2 and angle >1.5 (ish) you will get closed circular particles
//within this regime if speed > distance these aparticles move
//if these particles collide the more massive one will "absorb" the smaller

//CONTROLS
//pressing "s" toggles the image save function
//pressing "r" restarts the simulation

//enjoy!

//--------------------------------------------------------------
void ofApp::setup(){
	setupShaders(); //setting up shaders, 
	setupParams();  //setting up simulation params
	saving = false; //setting up saving conditional
}

//--------------------------------------------------------------
void ofApp::update() {
	updateAgents();		//dispatching simulation shader
	updatePheromones();	//dispatching diffusion shader
}

//--------------------------------------------------------------
void ofApp::draw(){
	showFPS();  //performance
	pheromoneIntensityTexture.draw(0, 0);  //drawing our texture buffer
	gui.draw(); //drawing GUI
	if (saving && ofGetFrameNum()%20==0) { 
		saveFrame(); //saving images
	}
}

//SETUP FUNCTIONS
//--------------------------------------------------------------
void ofApp::setupShaders() {
	//linking shaders
	simulation.setupShaderFromFile(GL_COMPUTE_SHADER, "simulation.glsl"); //simulations the stigmergy Agents
	simulation.linkProgram();
	diffusion.setupShaderFromFile(GL_COMPUTE_SHADER, "Diffusion.glsl");  //diffuses and decays pheremones
	diffusion.linkProgram();

	//initialising particle vector with particles
	particles.resize(1024*16*1024); //try and keep power of 2
	for (auto & particle : particles) {
		particle.pos = glm::vec3(ofGetWidth()*(0.5+(-0.2+ofRandom(0,0.4))), ofGetHeight()*(0.5 + (-0.2 + ofRandom(0, 0.4))),0);
		while (ofDist(particle.pos.x, particle.pos.y, ofGetWidth()*0.5, ofGetHeight()*0.5) > 0.2*ofGetWidth()) {
			particle.pos = glm::vec3(ofGetWidth()*(0.5 + (-0.2 + ofRandom(0, 0.4))), ofGetHeight()*(0.5 + (-0.2 + ofRandom(0, 0.4))), 0);
		}
		particle.heading = ofRandom(0, 2 * PI);
	}
	//initialising pheremone array with zero values
	for (int x = 0; x < W; x++) {
		for (int y = 0; y < H; y++) {
			int idx = x + y * W; //2D->1D
			pheremonesCPU[idx] = 0.0;
		}
	}

	//allocating buffer objects
	auto arraySize = W * H * sizeof(float);
	pheremones.allocate(arraySize, pheremonesCPU, GL_STATIC_DRAW);		//storing pheremone intensities
	pheremonesBack.allocate(arraySize, pheremonesCPU, GL_STATIC_DRAW);  //to allow for diffusion
	pheremonesClear.allocate(arraySize, pheremonesCPU, GL_STATIC_DRAW); //for resetting 
	particlesBuffer.allocate(particles, GL_DYNAMIC_DRAW);				//storing particle info
	particlesBufferClear.allocate(particles, GL_STATIC_DRAW);			//for restarting

	//binding buffers so GPU knows whats what
	//note we dont bind the "Clear" buffers as they are not referenced in the shader code
	particlesBuffer.bindBase(GL_SHADER_STORAGE_BUFFER, 0);
	pheremones.bindBase(GL_SHADER_STORAGE_BUFFER, 1);
	pheremonesBack.bindBase(GL_SHADER_STORAGE_BUFFER, 2);

	//allocating and binding display texture which we use to visualise GPU results
	pheromoneIntensityTexture.allocate(W, H, GL_RGBA8);
	pheromoneIntensityTexture.bindAsImage(3, GL_WRITE_ONLY);

}

void ofApp::setupParams() {
	//setting up GUI and simulation params
	gui.setup();
	//setting up agent params
	agentSettings.setName("Agent params");
	agentSettings.add(maxSpeed.set("maxSpeed", 2.7, 0, 20)); 			//step size of particles
	agentSettings.add(turningSpeed.set("turningSpeed", 1.0, 0, 3.141)); //angular step size
	agentSettings.add(sensorAngle.set("sensorAngle", 0.3, 0, 3.141));   //angle at which particles checks phermones
	agentSettings.add(sensorDistance.set("sensorDistance", 10, 1, 25)); //distance at when particle checks phermones
	agentSettings.add(sensorSize.set("sensorSize", 1, 0, 5));			//kernel size of pheremone check
	agentSettings.add(densitySpeed.set("densitySpeed", 0,0,1));
	agentSettings.add(baseMulti.set("baseMulti", 0.05, 0.001, 0.1));
	agentSettings.add(densityMulti.set("densityMulti", 0.01, 0.0001, 0.05));
	//setting up pheromone params
	pheromoneSettings.setName("Pheromone params");
	pheromoneSettings.add(decayWeight.set("decayWeight", 0.5, 0, 1));		  //value at which all pheromones decay
	pheromoneSettings.add(diffusionWeight.set("diffusionWeight", 0.1, 0, 1)); //value at which all pheromones diffuse
	pheromoneSettings.add(colouring.set("colouring", 0, 0, 1));
	//adding to GUI
	gui.add(agentSettings);
	gui.add(pheromoneSettings);

}


//SET UP FUNCTIONS
//---------------------------------------------------------------------------
void ofApp::maxSetup() {

	//params
	sampleRate = 44100;
	int initialBufferSize = 512;

	//defining arrays
	lAudioIn = new float[initialBufferSize];
	rAudioIn = new float[initialBufferSize];
	memset(lAudioIn, 0, initialBufferSize * sizeof(float));
	memset(rAudioIn, 0, initialBufferSize * sizeof(float));


	//fft params
	fftSize = 1024 * 4; //larger value for more resolution when sampled
	mfft.setup(fftSize, 512, 256);
	nAverages = 12;
	oct.setup(sampleRate, fftSize / 2, nAverages);

	//maxi setup
	ofxMaxiSettings::setup(sampleRate, 2, initialBufferSize);


	//oF sound settings
	ofSoundStreamSettings settings;
	soundStream.printDeviceList();
	auto devices = soundStream.getMatchingDevices("default", ofSoundDevice::Api::MS_WASAPI);
	if (!devices.empty()) {
		settings.setInDevice(devices[0]);
	}

	settings.setInListener(this);
	settings.setOutListener(this);
	settings.sampleRate = sampleRate;
	settings.numOutputChannels = 2;
	settings.numInputChannels = 2;
	settings.bufferSize = initialBufferSize;
	settings.setApi(ofSoundDevice::Api::MS_WASAPI);

	soundStream.setup(settings);
}



//UPDATE FUNCTIONS
//-------------------------------------------------------------
void ofApp::updateAgents() {
	//starting shader
	simulation.begin();
	//passing uniforms
	simulation.setUniforms(agentSettings);
	simulation.setUniform1i("W", W);
	simulation.setUniform1i("H", H);
	simulation.setUniform1i("numParticles", particles.size());
	//dispatching
	simulation.dispatchCompute(512, 1, 1); //splitting particles into 512 work groups (much faser than 1024 for some reason)
	//ending shader
	simulation.end();
}

void ofApp::updatePheromones() {
	//starting shader
	diffusion.begin();
	//passing uniforms
	diffusion.setUniforms(pheromoneSettings);
	diffusion.setUniform1i("W", W);
	diffusion.setUniform1i("H", H);
	//dispatching
	diffusion.dispatchCompute(W / 20, H / 20, 1); //splitting diffusion into work groups of size 20*20
	//ending shader
	diffusion.end();
	//diffusion shader sets diffused values to pheremoneBack buffer
	//copying those to main pheremone buffer for next frame
	pheremonesBack.copyTo(pheremones);
}

//Helper Functions
//--------------------------------------------------------------
void ofApp::showFPS() {
	ofSetColor(255);					 //setting white
	std::stringstream strm;
	strm << "fps: " << ofGetFrameRate(); //frame rate to sting stream
	ofSetWindowTitle(strm.str());
}

void ofApp::saveFrame() {
	//reading texture data into an oF image for saving
	pheromoneIntensityTexture.readToPixels(pixels);
	image.setFromPixels(pixels);
	image.save(ofToString(c) + ".png"); //saving with sequential names
	c++;
}

void ofApp::keyPressed(int key){
	//reseting the sim by copying the stored default values to the main buffers
	if (key == 'r') {
		particlesBufferClear.copyTo(particlesBuffer);
		pheremonesClear.copyTo(pheremones);
		pheremonesClear.copyTo(pheremonesBack);
	}
	//changing saving conditional
	if (key == 's') {
		saving = !saving;
	}
}




//audio functions
//--------------------------------------------------------------
void ofApp::audioOut(ofSoundBuffer& output) {
	//code adapted from maxiFeatureExtraction
	int outChannels = output.getNumChannels();
	for (int i = 0; i < output.getNumFrames(); ++i) {
		wave = lAudioIn[i];

		//get fft and octave

		if (mfft.process(wave)) {
			mfft.magsToDB();
			oct.calculate(mfft.magnitudesDB);

			float binFreq = 44100.0 / fftSize;
			float sumFreqs = 0;
			float sumMags = 0;
			float maxFreq = 0;
			int maxBin = 0;

			for (int i = 0; i < fftSize / 2; i++) {
				sumFreqs += (binFreq * i) * mfft.magnitudes[i];
				sumMags += mfft.magnitudes[i];
				if (mfft.magnitudes[i] > maxFreq) {
					maxFreq = mfft.magnitudes[i];
					maxBin = i;
				}
			}
			centroid = sumFreqs / sumMags;
			peakFreq = (float)maxBin * (44100.0 / fftSize);

		}

		//no output
		output[i * outChannels] = 0;
		output[i * outChannels + 1] = 0;
	}

}


void ofApp::audioIn(ofSoundBuffer& input) {
	//code adapted from maxiFeatureExtraction

	//getting audio in and calculating RMS
	float sum = 0;
	float count = 0;
	for (int i = 0; i < input.getNumFrames(); i++) {
		lAudioIn[i] = input[i * 2];
		rAudioIn[i] = input[i * 2 + 1];
		float sqr = input[i * 2] * input[i * 2];
		if (!isinf(sqr)) {
			sum += sqr * 1.0;
			count++;
		}
	}
	RMS = sqrt(sum / (count * 0.5));

	//calculating time averaged RMS
	if (timeAverageRMS.size() < int(10)) {
		if (!isinf(RMS) && !isnan(RMS)) {
			timeAverageRMS.push_back(RMS);
		}
	}
	if (timeAverageRMS.size() == int(10)) {
		timeAverageRMS.erase(timeAverageRMS.begin());
	}

	//summing up values in time average
	float timeAverageLong = 0.0;
	int c = 0;
	for (auto& n : timeAverage) {
		if (!isinf(n) && !isnan(n)) { //quick valid check
			timeAverageLong += n;
			c++;
		}
	}

	RMS = timeAverageLong / c;
}




//Calculations
//---------------------------------------------------------------------------

ofApp::Result ofApp::getAudioFeatures() {
	//extracts average FFT mag in the permitted freq domain
	//min and max FFT bins (for visual)
	//bin freq for displays
	//pitch centroid

	int size = 0;
	float sum = 0;
	int minBin = fftSize * 2;
	int maxBin = 0;
	float binFreq = float(sampleRate) / fftSize;

	//getting FFT mags and bin info
	for (int i = 0; i < fftSize / 2; ++i) {
		float freq = float(i)*binFreq; //freq to bin index
		if (freq > minFreq && freq < maxFreq) { //domain check
			float val = mfft.magnitudes[i];
			if (!isinf(val) && !isnan(val)) { //forcing only valid values
				size++;
				sum += val;
				if (i > maxBin) {
					maxBin = i;
				}
				if (i < minBin) {
					minBin = i;
				}
			}

		}
	}

	//getting pitch info
	float pitchCentroid = 0;
	float freqSum = 0;
	float valSum = 0;
	for (int i = 0; i < oct.nAverages; i++) {
		float val = oct.averages[i];
		ofLog() << " vak " << val * 10;
		if (!isinf(val) && !isnan(val) && val > 0.1) {
			freqSum += i;
			valSum += val;
		}
	}
	pitchCentroid = freqSum / valSum;
	return Result{ sum / size, minBin, maxBin, binFreq, pitchCentroid };

}




//function to calculate the time average of FFT values
float ofApp::fftLongAvg(float timeAverageShort) {
	//filling time average vector, fixed length (roughly last second, can extend)
	if (timeAverage.size() < int(timeAverageLength)) {
		if (!isinf(timeAverageShort) && !isnan(timeAverageShort)) {
			timeAverage.push_back(timeAverageShort);
		}
	}
	if (timeAverage.size() == int(timeAverageLength)) {
		timeAverage.erase(timeAverage.begin());
	}

	//summing up values in time average
	float timeAverageLong = 0.0;
	int c = 0;
	for (auto& n : timeAverage) {
		if (!isinf(n) && !isnan(n)) { //quick valid check
			timeAverageLong += n;
			c++;
		}
	}

	return timeAverageLong / c;
}

//function to calculate time average of pitch values
//should probably make this into the above funciton with some more generalisation
float ofApp::maxPitchBinLongAverage(float maxPitchBin) {
	//filling time average vector, fixed length (roughly last second, can extend)
	if (timeAveragePitch.size() < int(pitchAverageLength)) {
		if (!isinf(maxPitchBin) && !isnan(maxPitchBin)) {
			timeAveragePitch.push_back(maxPitchBin);
		}
	}
	if (timeAveragePitch.size() == int(pitchAverageLength)) {
		timeAveragePitch.erase(timeAveragePitch.begin());
	}

	//summing up values in time average
	float pitchAverageLong = 0.0;
	int c = 0;
	for (auto& n : timeAveragePitch) {
		if (!isinf(n) && !isnan(n)) { //quick valid check
			pitchAverageLong += n;
			c++;
		}
	}

	return pitchAverageLong / c;
}

//beat detection visuals and logic
bool ofApp::beatDetctor(float timeAverageShort, float timeAverageLong) {

	int gap = (horizWidth - 80) / 3;
	//this bar is the minimum power a frequency must have to be considred 
	//shown in red
	ofSetColor(255, 0, 0, 255);
	ofDrawRectangle(horizOffset, vertOffset + 50 - detectThreshold * 50, 20, 3);

	//this bar is the current "average" from the previous samples
	//shown in green
	ofSetColor(0, 255, 0, 255);
	ofDrawRectangle(horizOffset + gap + 20, vertOffset + 50 - timeAverageLong * 50, 20, 3);

	//this bar is the threshold for a beat to be considered
	//shown in white
	ofSetColor(255, 255, 255, 255);
	ofDrawRectangle(horizOffset + 2 * gap + 40, vertOffset + 50 - timeAverageLong * 50 * DetectMultiplier, 20, 3);

	//this bar is the current average value in the allowed frequencies
	// shown in blue
	ofSetColor(0, 0, 255, 255);
	ofDrawRectangle(horizOffset + 3 * gap + 60, vertOffset + 50 - timeAverageShort * 50, 20, 3);


	//beat condition
	if (timeAverageShort > DetectMultiplier*timeAverageLong && timeAverageShort > detectThreshold) {
		return true;
	}
	else {
		return false;
	}

}
//----------------------------------------------------------------


//drawing functions
//---------------------------------------------------------------------------------
void ofApp::drawFFT(int minBin, int maxBin, float binFreq, bool onBeat) {
	//visualing FFT (adapted from maxIFeature)
	if (onBeat) {
		ofSetColor(255);
	}
	else {
		ofSetColor(255, 0, 0);
	}

	ofPolyline line;
	int cc = 0;
	float xinc = horizWidth / (maxBin - minBin);
	for (int i = minBin; i < maxBin; ++i) {
		float freq = float(i)*binFreq;
		if (freq > minFreq && freq < maxFreq) {
			cc++;
			float height = mfft.magnitudes[i] * 50;
			line.addVertex(ofPoint(horizOffset + (cc * xinc), vertOffset - height));
		}
	}
	line.draw();

}

//visualins octaves (from maxi feature)
void ofApp::drawOct() {
	ofSetColor(255, 0, 255, 200);
	float xinc = horizWidth / oct.nAverages;
	float maxPitchVal = 0;
	for (int i = 0; i < oct.nAverages; i++) {
		float val = oct.averages[i];
		if (!isinf(val) && !isnan(val)) {
			float height = val / 20.0 * 50;
			ofDrawRectangle(horizOffset + (i * xinc), vertOffset - height + 150, 2, height);
		}
	}
}


void ofApp::drawInfo(float timeAverageShort) {
	//displaying infomation adpated from maxi feature
	ofSetColor(255);
	myfontBig.drawString("VISUALISER", horizOffset, 50);

	char avgString[255]; // an array of chars
	sprintf(avgString, "AVG: %.4f", timeAverageShort);
	myfont.drawString(avgString, horizOffset, vertOffset + 200);

	char minFreqStr[255];
	sprintf(minFreqStr, "minFreq: %.0f", float(minFreq));
	myfont.drawString(minFreqStr, horizOffset, vertOffset + 230);

	char maxFreqStr[255];
	sprintf(maxFreqStr, "maxFreq: %.0f", float(maxFreq));
	myfont.drawString(maxFreqStr, horizOffset, vertOffset + 260);


	char peakString[255];
	sprintf(peakString, "Peak Freq: %.2f", peakFreq);
	myfont.drawString(peakString, horizOffset, vertOffset + 290);

	char centroidString[255];
	sprintf(centroidString, "Spec Cent: %.2f", centroid);
	myfont.drawString(centroidString, horizOffset, vertOffset + 320);

	char pitchString[255];
	sprintf(pitchString, "Pitch Cent: %.2f", pitchValue);
	myfont.drawString(pitchString, horizOffset, vertOffset + 350);


	char rmsString[255];
	sprintf(rmsString, "RMS: %.2f", RMS);
	myfont.drawString(rmsString, horizOffset, vertOffset + 380);
}

