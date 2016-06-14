import qi

class Pepper_speech(object):
    def __init__(self):

        self.ip = "10.65.34.43"
        self.port = 9559
        self.session = qi.Session()

        try:
            self.session.connect("tcp://" + self.ip + ":" + str(self.port))
        except RuntimeError:
            print ( "Can't connect to Naoqi at ip \"" + self.ip + "\" on port " + str(self.port) +".\n"
                    "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)

        self.ALTextToSpeech = self.session.service('ALTextToSpeech')

    def tts(text):
        self.ALTextToSpeech.say(text)

app = App()
app.tts("bonjour")
