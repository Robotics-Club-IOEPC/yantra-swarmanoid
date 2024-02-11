#include <WiFi.h>
#include "PubSubClient.h"

//watch monster moterdriver pinout  for ain1, ain2, bin1, bin2, etc
//enable pin should be high with 3.3v
#define m1_ain1 16 //left
#define m1_bin2 3   //right
#define m1_bin1 18   //left
#define m1_ain2 19   //right
#define m1_pwma 23    //(left)
#define m1_pwmb 22    //(right) 

#define Relay_1 25
#define Relay_2 26
#define limit_1 12
#define limit_2 4

int flag_open = 0;
int flag_close = 0;


int left_min_pwm_forward = 105;
int left_min_pwm_backward = 85;
int right_min_pwm_forward = 105;
int right_min_pwm_backward = 90;
WiFiClient espClient22;
PubSubClient client(espClient22);

// Replace with your network credentials (STATION)
const char* ssid = "AI/ML";
const char* password = "ACES@1234";

const char* mqtt_server = "192.168.1.117";

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
  Serial.begin(9600);
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
  pinMode(Relay_1, OUTPUT);
  pinMode(Relay_2, OUTPUT);
  pinMode(limit_1, INPUT_PULLUP);
  pinMode(limit_2, INPUT_PULLUP);
  
  
  digitalWrite(Relay_1, HIGH);
  digitalWrite(Relay_2, HIGH);
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
      int pwm_value = messageInfo.toFloat();
      if(pwm_value != 0)
        pwm_value = map(left_min_pwm_forward,255,0,255,pwm_value);
      analogWrite(m1_pwma,pwm_value);
      digitalWrite(m1_ain1, LOW);
      digitalWrite(m1_bin1, HIGH);
      Serial.print("robot7_left_forward : ");
      Serial.println(pwm_value);
  }
  if(topic=="/robot7_right_forward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toFloat();
      if(pwm_value != 0)
        pwm_value = map(right_min_pwm_forward,255,0,255,pwm_value);
      analogWrite(m1_pwmb,pwm_value);
      digitalWrite(m1_ain2, HIGH);
      digitalWrite(m1_bin2, LOW);
      Serial.print("robot7_right_forward : ");
      Serial.println(pwm_value);
  }
  if(topic =="/robot7_gripper_close"){
    Serial.println(digitalRead(limit_1));
    if(!digitalRead(limit_1)){
      Serial.println("Gripper_closed");
    }
    else{
        flag_close = 1;     
    }
    }

  // }
  if(topic =="/robot7_gripper_open"){
    if(!digitalRead(limit_2)){
      Serial.println("Gripper_open");
    }
    else{
        flag_open = 1;
      }

  }
  if(topic=="/robot7_left_backward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toFloat();
      if(pwm_value != 0)
        pwm_value = map(left_min_pwm_backward,255,0,255,pwm_value);
      analogWrite(m1_pwma,pwm_value);
      digitalWrite(m1_ain1, HIGH);
      digitalWrite(m1_bin1, LOW);
      Serial.print("robot7_left_backward : ");
      Serial.println(pwm_value);
  }
  if(topic=="/robot7_right_backward"){
      // Serial.print("Displaying message to users");
      // if(messageInfo == "allert"){
      //  len = messageInfo.length();
      int pwm_value = messageInfo.toFloat();
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
      client.subscribe("/robot7_gripper_open");
      client.subscribe("/robot7_gripper_close");
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
  if(flag_open == 1){
    // Serial.println(digitalRead(limit_2));

    if(digitalRead(limit_2)){
      Serial.println("#");
      digitalWrite(Relay_1,HIGH);
      digitalWrite(Relay_2,LOW);
    }
    else{
      digitalWrite(Relay_1,HIGH);
      digitalWrite(Relay_2,HIGH);
      flag_open = 0;
    }
  }
  // Serial.println(digitalRead(limit_1));
  if(flag_close == 1){
    Serial.print("close");
    Serial.println(digitalRead(limit_1));

    if(digitalRead(limit_1)){
      digitalWrite(Relay_1,LOW);
      digitalWrite(Relay_2,HIGH);
    }
    else{
      digitalWrite(Relay_1,HIGH);
      digitalWrite(Relay_2,HIGH);
      flag_close = 0;
    }
  }  
if(!client.loop()){
    client.connect("ESP32Client21");
      }
}







