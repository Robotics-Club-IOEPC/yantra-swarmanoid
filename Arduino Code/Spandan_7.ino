#include <WiFi.h>
#include "PubSubClient.h"

//watch monster moterdriver pinout  for ain1, ain2, bin1, bin2, etc
//enable pin should be high with 3.3v
#define m1_ain1 16 //left
#define m1_bin2 22   //right
#define m1_bin1 18   //left
#define m1_ain2 4   //right
#define m1_pwma 23    //(left)
#define m1_pwmb 17    //(right) 

int left_min_pwm_forward = 160;
int left_min_pwm_backward = 150;
int right_min_pwm_forward = 160;
int right_min_pwm_backward = 130;
WiFiClient espClient22;
PubSubClient client(espClient22);

// Replace with your network credentials (STATION)
const char* ssid = "Robotics";
const char* password = "Robotics123";

const char* mqtt_server = "192.168.1.97";

void initWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print('.');
    delay(1000);
  }
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  initWiFi();
  Serial.print("RRSI: ");
  Serial.println(WiFi.RSSI());
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  pinMode(m1_ain1, OUTPUT);
  pinMode(m1_ain2, OUTPUT);
  pinMode(m1_bin1, OUTPUT);
  pinMode(m1_bin2, OUTPUT);
  pinMode(m1_pwma, OUTPUT);
  pinMode(m1_pwmb, OUTPUT);
  
  // digitalWrite(m1_pwma, HIGH);
  // digitalWrite(m1_pwmb, HIGH);
}

void callback(String topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  
  String messageInfo;
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    messageInfo += (char)message[i];
  }
  Serial.println();

  // If a message is received on the topic room/lamp, you check if the message is either on or off. Turns the lamp GPIO according to the message
  if(topic=="/robot7_left_forward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toInt();
      if(pwm_value != 0)
        pwm_value = map(left_min_pwm_forward,255,0,255,pwm_value);
      analogWrite(m1_pwma,pwm_value);
      digitalWrite(m1_ain1, HIGH);
      digitalWrite(m1_bin1, LOW);
      Serial.print("robot7_left_forward : ");
      Serial.println(pwm_value);
  }
  if(topic=="/robot7_right_forward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toInt();
      if(pwm_value != 0)
        pwm_value = map(right_min_pwm_forward,255,0,255,pwm_value);
      analogWrite(m1_pwmb,pwm_value);
      digitalWrite(m1_ain2, HIGH);
      digitalWrite(m1_bin2, LOW);
      Serial.print("robot7_right_forward : ");
      Serial.println(pwm_value);
  }
  if(topic=="/robot7_left_backward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toInt();
      if(pwm_value != 0)
        pwm_value = map(left_min_pwm_backward,255,0,255,pwm_value);
      analogWrite(m1_pwma,pwm_value);
      digitalWrite(m1_ain1, LOW);
      digitalWrite(m1_bin1, HIGH);
      Serial.print("robot7_left_backward : ");
      Serial.println(pwm_value);
  }
  if(topic=="/robot7_right_backward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toInt();
      if(pwm_value != 0)
        pwm_value = map(right_min_pwm_backward,255,0,255,pwm_value);
      analogWrite(m1_pwmb,pwm_value);
      digitalWrite(m1_ain2, LOW);
      digitalWrite(m1_bin2, HIGH);
      Serial.print("robot7_right_backward : ");
      Serial.println(pwm_value);
  }
  // if(topic=="/robot6_left_forward"){
  //     // Serial.print("Displaying message to users");
  //     // if(messageInfo == "allert"){
  //     //  len = messageInfo.length();
      
  //     int pwm_value = messageInfo.toInt();
  //     if(pwm_value != 0)
  //       pwm_value = map(left_min_pwm_forward,255,0,255,pwm_value);
  //     analogWrite(m1_pwma,pwm_value);
  //     digitalWrite(m1_bin1, HIGH);
  //     digitalWrite(m1_ain1, LOW);
  //     Serial.print("robot6_left_forward : ");
  //     Serial.println(pwm_value);
  // }
  // if(topic=="/robot6_right_forward"){
  //     // Serial.print("Displaying message to users");
  //     // if(messageInfo == "allert"){
  //     //  len = messageInfo.length();
  //     int pwm_value = messageInfo.toInt();
  //     if(pwm_value != 0)
  //       pwm_value = map(right_min_pwm_forward,255,0,255,pwm_value);
  //     analogWrite(m1_pwmb,pwm_value);
  //     digitalWrite(m1_ain2, HIGH);
  //     digitalWrite(m1_bin2, LOW);
  //     Serial.print("robot6_right_forward : ");
  //     Serial.println(pwm_value);
  // }
  // if(topic=="/robot6_left_backward"){
  //     // Serial.print("Displaying message to users");
  //     // if(messageInfo == "allert"){
  //     //  len = messageInfo.length();
  //     int pwm_value = messageInfo.toInt();
  //     if(pwm_value != 0)
  //       pwm_value = map(left_min_pwm_backward,255,0,255,pwm_value);
  //     analogWrite(m1_pwma,pwm_value);
  //     digitalWrite(m1_bin1, LOW);
  //     digitalWrite(m1_ain1, HIGH);
  //     Serial.print("robot6_left_backward : ");
  //     Serial.println(pwm_value);
  // }
  // if(topic=="/robot6_right_backward"){
  //     Serial.print("Displaying message to users");
  //     // if(messageInfo == "allert"){
  //     //  len = messageInfo.length();
  //     int pwm_value = messageInfo.toInt();
  //     if(pwm_value != 0)
  //       pwm_value = map(right_min_pwm_backward,255,0,255,pwm_value);
  //     analogWrite(m1_pwmb,pwm_value);
  //     digitalWrite(m1_ain2, LOW);
  //     digitalWrite(m1_bin2, HIGH);
  //     Serial.print("robot6_right_backward : ");
  //     Serial.println(pwm_value);
  // }
  Serial.println();
}
// void backward()
// {
//   digitalWrite(m1_ain1, LOW);
//   digitalWrite(m1_bin1, LOW);
//   digitalWrite(m1_ain2, HIGH);
//   digitalWrite(m1_bin2, HIGH);
//   Serial.println("forward");
// }

// void forward()
// {
//   digitalWrite(m1_ain1, HIGH);
//   digitalWrite(m1_bin1, HIGH);
//   digitalWrite(m1_ain2, LOW);
//   digitalWrite(m1_bin2, LOW);
//   Serial.println("backward");
// }

// void left()
// {
//   digitalWrite(m1_ain1, LOW);
//   digitalWrite(m1_bin1, HIGH);
//   digitalWrite(m1_ain2, HIGH);
//   digitalWrite(m1_bin2, LOW);
//   Serial.println("left");
// }

// void right()
// {
//   digitalWrite(m1_ain1, HIGH);
//   digitalWrite(m1_bin1, LOW);
//   digitalWrite(m1_ain2, LOW);
//   digitalWrite(m1_bin2, HIGH);
//   Serial.println("right");
// }

// void stop()
// {
//   digitalWrite(m1_ain1, LOW);
//   digitalWrite(m1_bin1, LOW);
//   digitalWrite(m1_ain2, LOW);
//   digitalWrite(m1_bin2, LOW);
//   Serial.println("stop");
// }

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    
    if (client.connect("ESP32Client21")) {
      Serial.println("connected");  
      // Subscribe or resubscribe to a topic
      // You can subscribe to more topics (to control more LEDs in this example)
      // client.subscribe("/robot7_left_forward");
      // client.subscribe("/robot7_right_forward");
      // client.subscribe("/robot7_left_backward");
      // client.subscribe("/robot7_right_backward");
      client.subscribe("/robot7_left_forward");
      client.subscribe("/robot7_right_forward");
      client.subscribe("/robot7_left_backward");
      client.subscribe("/robot7_right_backward");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }

if(!client.loop()){
    client.connect("ESP32Client21");
      }
}







