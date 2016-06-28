/*eslint-env jquery, browser*/
/*globals webkitSpeechRecognition */
'use strict';

var clientid = -1;

var url = ''; // Server IP address ('' for localhost)

// initial load
$(document).ready(function() {

	clientid = Math.floor(Math.random() * 10000 + 1);

	$.post(url+'/start?id='+clientid);

	waitForServerInput();
});

function waitForServerInput() {
	$.post(url+'/wait?id='+clientid).done(function(data) {
		waitForServerInput();
		if (data !== ""){
			// Display photos according to hyperlink
			//$('div.images').html('<img src="' + data + '" class="w3-border w3-padding-4 w3-padding-tiny" alt="photo">');

			if (data.indexOf("L") === 0){
				var imgPath = "face_database/";
				var suffix  = ".png"
			}

			else if (data.indexOf("R") === 0){
				var imgPath = "face_database_for_oxford/";
				var suffix  = ".jpg"
			}

			var start = data.indexOf(imgPath) + imgPath.length;
			var end   = data.indexOf(suffix) - 2;

			var link = data.substring(2);
			var name = data.substring(start, end);
			var firstLetter = name[0].toUpperCase(); 

			$('h1.bonjour').html("Bonjour " + firstLetter.concat(name.substring(1)) + " !");
			$('center.images').append('&emsp;<img src="' + link + '" class="w3-border w3-padding-4 w3-padding-tiny" alt="photo"  style="width:128px;">');
		}
	});
}
