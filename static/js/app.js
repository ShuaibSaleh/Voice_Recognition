// webkitURL is deprecated but nevertheless
URL = window.URL || window.webkitURL;

var gumStream; 						// Stream from getUserMedia()
var rec; 							// Recorder.js object
var input; 							// MediaStreamAudioSourceNode we'll be recording

// Chips for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext // Audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var recordedAudio = document.getElementById("recordedAudio");
var showposter = document.getElementById("show");
// Add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

function startRecording() {
	console.log("recordButton clicked");

	
	// Simple constraints object
    var constraints = { audio: true, video:false }

	// Disable the record button until we get a success or fail from getUserMedia() 
	recordButton.disabled = true;
	stopButton.disabled = false;

    // We're using the standard promise based getUserMedia() 
	
	navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
		console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

		
		// Create an audio context after getUserMedia is called
			
		audioContext = new AudioContext();

		// Assign to gumStream for later use  
		gumStream = stream;
		
		// Use the stream 
		input = audioContext.createMediaStreamSource(stream);

		// Create the Recorder object and configure to record mono sound (1 channel)
		rec = new Recorder(input,{numChannels:1});

		// Start the recording process
		rec.record();
		console.log("Recording started");
	}).catch(function(err) {
	  	// Enable the record button if getUserMedia() fails
    	recordButton.disabled = false;
    	stopButton.disabled = true;
		});
}


function stopRecording() {
	console.log("stopButton clicked");

	// Disable the stop button, enable the record too allow for new recordings
	stopButton.disabled = true;
	recordButton.disabled = false;
	// Stop the recording
	rec.stop();
	// Stop microphone access
	gumStream.getAudioTracks()[0].stop();

	// Create the wav blob and pass it on to createProcessAudio
	rec.exportWAV(createProcessAudio);
}
function images(){
	var timestamp = new Date().getTime();     
	var bc = document.getElementById("barchart"); 
	bc.src = "/static/img/barchart.png?t=" + timestamp;    
	bc.classList.remove("hidden");
	var sig = document.getElementById("signal"); 
	sig.src = "/static/img/signal.png?t=" + timestamp;    
	sig.classList.remove("hidden");
	var spec = document.getElementById("spec"); 
	spec.src = "/static/img/spectrogram.png?t=" + timestamp;    
	spec.classList.remove("hidden");
};
function createProcessAudio(blob) {
	
	var url = URL.createObjectURL(blob);

	//name of .wav file to use during upload and download (without extendion)
	var filename = new Date().toISOString();

	//add controls to the <audio> element
	recordedAudio.controls = true;
	recordedAudio.src = url;
	recordedAudio.autoplay = true
	// recordedAudio.autoplay;

	// Send data to the backend (python)
	var xhr=new XMLHttpRequest();
	var fd=new FormData();
	fd.append("audio_data",blob, filename);
	xhr.onreadystatechange = function() {
		if (xhr.status == 200) {
			
			document.getElementById("formats").innerHTML=" "+xhr.responseText
			document.getElementById("barchart").innerHTML = images()
		}
	  };
	xhr.open("POST","/",true);
	xhr.send(fd);

	
	
};

const toggleTo2 = document.getElementById("recordButton");
const toggleTo1 = document.getElementById("stopButton");

const hide = el => el.style.setProperty("display", "none");
const show = el => el.style.setProperty("display", "block");

hide(toggleTo1);

toggleTo2.addEventListener("click", () => {
  hide(toggleTo2);
  show(toggleTo1);
});

toggleTo1.addEventListener("click", () => {
  hide(toggleTo1);
  show(toggleTo2);
});