import time
import sys
import qi


class App(object):
    def __init__(self):

        self.ip = "169.254.16.208"
        self.port = 9559

        self.session = qi.Session()

        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ( "Can't connect to Naoqi at ip \"" + self.ip + "\" on port " + str(self.port) +".\n"
                    "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)

        self.ALAudioDevice = self.session.service('ALAudioDevice')
        self.ALTextToSpeech = self.session.service('ALTextToSpeech')
    def run(self):
        text = 'bonsoir'
        self.ALTextToSpeech.say(text)
        time.sleep(0.1)
        self.ALAudioDevice.playSine(1000, 100, 0, 0.5)

        self.ALAudioDevice.startMicrophonesRecording('pepper.local/recordings/microphones/test.wav')
        time.sleep(5)
        self.ALAudioDevice.stopMicrophonesRecording()

app = App()
app.run()
