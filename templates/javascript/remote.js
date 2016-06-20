/*eslint-env jquery, browser*/
/*globals webkitSpeechRecognition */
'use strict';

var audioQueue = [];
var audio = $('.audio').get(0);

var isChrome = !!window.chrome && !!window.chrome.webstore;

var $dialogsLoading = $('.dialogs-loading');

// conversation nodes
var $conversation = $('.conversation-container');
var $userInput = $('.user-input');

var clientid = -1;

var url = ''; // Server IP address


// initial load
$(document).ready(function() {

	clientid = Math.floor(Math.random() * 1000000000 + 1);

	$.post(url+'/chat?text=START&id='+clientid); //TODO: new

	$('.audio').on('ended', function() {checkAudioQueue();});

	$('.listen-btn').click(listenAndWrite);

	$('.input-btn').click(conductConversation);
  $userInput.keyup(function(event){
      if(event.keyCode === 13) {
          conductConversation();
      }
  });

	$userInput.focus();

	/*for (var i=0;i<5;i++) {
		waitForServerInput();
	}*/
	waitForServerInput(); //TODO: new

	//$.post(url+'/chat?text=START&id='+clientid);
	if (isChrome) {
		// get the speech synthesis ready to avoid english accent on the first sentence
		var msg = new SpeechSynthesisUtterance();
		var voices = speechSynthesis.getVoices();
	}
});

function conductConversation() {
	stopAudio();

    var userIntentText = $userInput.val();
    $userInput.val('').focus();

	/*if($dialogsLoading.css('display') !== 'none') {
		displayHumanChat(userIntentText);
		displayBotChat("Patientez encore un peu s'il vous plaît.");
		$dialogsLoading.show();
		return;
	}

	$dialogsLoading.show();*/

	displayHumanChat(userIntentText);

	$.post(url+'/chat?text='+encodeURIComponent(userIntentText)+"&id="+clientid);
}

function waitForServerInput() {
	$.post(url+'/wait?id='+clientid).done(function(data) {
		waitForServerInput();
		if (data !== "RECONNECT" && data !== ""){
			if (data.indexOf("THINK") === 0) { //TODO: new
				var txt = data.substring(6);
				displayBotChat(txt);
				$dialogsLoading.show();
			}
			else if (data.indexOf("START") === 0) {
				$dialogsLoading.show();
			}
			else if (data.indexOf("DONE") === 0) {
				$dialogsLoading.hide();
			}
			else {
				displayBotChat(data);
				$dialogsLoading.hide();
			}
		}
	});
}

function displayBotChat(text) {

	$('<div class="bubble-watson"/>').html(text).appendTo($conversation);
	scrollToBottom();
	//startAudio(text); /* Put in comment because it's not necessary to use Text-To-Speech (Pepper will do it) */

}

function displayHumanChat(text) {

    $('<p class="bubble-human"/>').html(text)
        .appendTo($conversation);

    $('<div class="clear-float"/>')
        .appendTo($conversation);

    scrollToBottom();
}

function scrollToBottom (){
    $('body, html').animate({ scrollTop: $('body').height() + 'px' });
}

function startAudio(txt) {
	if (isChrome) {
		var chunkLength = 300;
	    var pattRegex = new RegExp('^[\\s\\S]{' + Math.floor(chunkLength / 2) + ',' + chunkLength + '}[.!?,]{1}|^[\\s\\S]{1,' + chunkLength + '}$|^[\\s\\S]{1,' + chunkLength + '} ');

		var arr = [];
	    while (txt.length > 0) {
	        arr.push(txt.match(pattRegex)[0]);
	        txt = txt.substring(arr[arr.length - 1].length);
	    }
	    $.each(arr, function () {
	        var msg = new SpeechSynthesisUtterance(this.trim());
	        var voices = speechSynthesis.getVoices();
			msg.voice = voices[7];
			msg.voiceURI = 'native';
			msg.volume = 1;
			msg.rate = 1.2;
			msg.pitch = 1;
			msg.lang = 'fr-FR';
	        window.speechSynthesis.speak(msg);
		});
	}
}

function stopAudio() {
	if (isChrome) {
		//Stop audio if users has already aswered to the question
		window.speechSynthesis.cancel();
	}
}

function listenAndWrite(){
	if (isChrome) {
		$.post(url+'/startListening?id='+clientid);
		stopAudio();

		var recognition = new webkitSpeechRecognition();
		recognition.lang = "fr-FR";
		//recognition.continuous = true;
		//recognition.interimResults = true;
		recognition.onresult = function(event) {
			var text = event.results[0][0].transcript
			//console.log(event);
			$dialogsLoading.hide();
			$userInput.val(event.results[0][0].transcript);
			conductConversation();
		}
		$dialogsLoading.show();
		recognition.start();
	}
}
