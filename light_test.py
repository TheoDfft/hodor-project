import uuid
import requests

class ToggleKasaLight():

    def __init__(self):
        self.USER_EMAIL = "duffaut.theo@gmail.com"
        self.USER_SECRET = "Td13571357"
        self.token = None
        self.deviceID = None

    def getKasaToken(self):
        payload = {
                "method": "login",
                "params": {
                    "appType": "Kasa_Android",
                    "cloudUserName": self.USER_EMAIL,
                    "cloudPassword": self.USER_SECRET,
                    "terminalUUID": str(uuid.uuid4())
                }
        }
        response = requests.post(url="https://wap.tplinkcloud.com/", json=payload)
        obj = response.json()
        self.token = obj["result"]["token"]

    def getKasaDeviceList(self):
        payload = {"method": "getDeviceList"}
        device_list = requests.post("https://wap.tplinkcloud.com?token={}".format(self.token), json=payload)
        # we can use [0]['devideId'] here as we only have one device
        self.deviceID = device_list.json()['result']['deviceList'][0]['deviceId']

    def modifyKasaDeviceState(self, deviceState):
        self.getKasaToken()
        self.getKasaDeviceList()

        payload = {
                "method": "passthrough",
                "params": {
                    "deviceId": self.deviceID,
                    "requestData":
                        '{\"system\":{\"set_relay_state\":{\"state\":' + str(deviceState) + '}}}'
                }
            }
        requests.post(url="https://use1-wap.tplinkcloud.com/?token={}".format(self.token), json=payload)
