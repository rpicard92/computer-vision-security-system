package com.example.computer.securitysystemapp;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import java.io.IOException;
import java.net.InetAddress;

import java.net.Socket;
import java.net.UnknownHostException;
import android.widget.Button;
import android.view.View;
import android.os.StrictMode;
import java.io.DataOutputStream;

import org.json.JSONException;
import org.json.JSONObject;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.net.*;

public class MainActivity extends AppCompatActivity {

    private Socket socket;
    private DataOutputStream dataOutputStream;
    private JSONObject jsonData;
    private InputStreamReader inputStreamReader;
    private BufferedReader bufferedReader;
    private String message;
    private ServerSocket SRVSOCKET;
    private static final int SERVER_PORT = 8889;
    private static final String SERVER_IP = "XXX.XXX.XXX.XXX";
    private static final int ANDROID_PORT = 8888;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // button handler
        final Button button = findViewById(R.id.button_id);
        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                // Code here executes on main thread after user presses button
                StrictMode.ThreadPolicy policy = new StrictMode.ThreadPolicy.Builder().permitAll().build();
                StrictMode.setThreadPolicy(policy);

                try {
                    // create new client socket
                    InetAddress serverAddr = InetAddress.getByName(SERVER_IP);
                    socket = new Socket(serverAddr, SERVER_PORT);

                    // make a data out stream
                    dataOutputStream = new DataOutputStream(socket.getOutputStream());


                    // create a jason object
                    jsonData = new JSONObject();
                    try {
                        jsonData.put("foo",1);
                    }
                    catch (JSONException e) {
                        int a = 1;
                    }

                    // send message to the server  // note: right now you can send any data
                    dataOutputStream.writeUTF(jsonData.toString());
                    System.out.println("MESS SENT....");

                    // close everything
                    dataOutputStream.flush();
                    dataOutputStream.close();
                    socket.close();

                    // create server socket
                    SRVSOCKET = new ServerSocket(ANDROID_PORT);
                    socket = SRVSOCKET.accept();

                    // wait for response
                    while(true) {
                        inputStreamReader = new InputStreamReader(socket.getInputStream());
                        bufferedReader = new BufferedReader(inputStreamReader);
                        message = bufferedReader.readLine();
                        if(message != null) {
                            System.out.println("MESSAGE FROM SERVER: " + message);
                            break;
                        }
                    }

                    // close everything
                    bufferedReader.close();
                    inputStreamReader.close();
                    socket.close();
                    SRVSOCKET.close();

                    // change the button text
                    button.setText(message);

                } catch (UnknownHostException e1) {
                    e1.printStackTrace();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        });


    }
}
