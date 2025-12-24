using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

/// <summary>
/// Unity-side TCP server + simple analytical IK for a UR3e-like 6-DOF arm.
/// - Listens on port 12345 for ASCII commands.
/// - Updates target position (px,py,pz) and orientation (rx,ry,rz) and solves IK each Update().
///
/// Protocol examples:
///   "left" / "right" / "up" / "down" / "forward" / "backward"
///   "rx 45"  (degrees)
///   "ry -30"
///   "rz 90"
/// </summary>
public class InverseKinematicsKeyboardControl : MonoBehaviour
{
    public Transform[] Joints;                 // 6 joint transforms in order
    public static double[] theta = new double[6];  // Joint angles (radians)

    // UR3e link lengths (meters). Adjust if your Unity model uses a different scale.
    private float L1 = 0.1519f;   // Base to shoulder
    private float L2 = 0.24365f;  // Shoulder to elbow
    private float L3 = 0.21325f;  // Elbow to wrist 1
    private float L4 = 0.0f;      // Wrist 1 to wrist 2 (simplified)
    private float L5 = 0.0f;      // Wrist 2 to wrist 3 (simplified)
    private float L6 = 0.0f;      // Tool length (simplified)

    private float C3;

    // Target position (Unity units) and orientation (degrees)
    public float px, py, pz;
    public float rx, ry, rz;

    public float positionStep = 20f; // Step size for discrete moves
    public float groundAngle = 0f;   // Example: angle relative to ground (deg)

    // Socket server
    private TcpListener tcpListener;
    private TcpClient tcpClient;
    private NetworkStream stream;
    private byte[] buffer = new byte[1024];

    void Start()
    {
        for (int i = 0; i < 6; i++) theta[i] = 0.0;

        // Example initial target
        px = 0f;
        py = -3.54f;
        pz = 7.479113f;

        StartSocketServer();
    }

    void Update()
    {
        // Read incoming TCP data if available
        if (tcpClient != null && tcpClient.Available > 0)
        {
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            string message = Encoding.ASCII.GetString(buffer, 0, bytesRead);
            HandleMessage(message);
        }

        CalculateIK();
        ApplyJointAngles();
    }

    private void StartSocketServer()
    {
        tcpListener = new TcpListener(IPAddress.Any, 12345);
        tcpListener.Start();
        Debug.Log("Server started on port 12345");

        tcpListener.BeginAcceptTcpClient(new AsyncCallback(OnClientConnect), null);
    }

    private void OnClientConnect(IAsyncResult ar)
    {
        tcpClient = tcpListener.EndAcceptTcpClient(ar);
        stream = tcpClient.GetStream();
        Debug.Log("Client connected");

        // Continue accepting other clients
        tcpListener.BeginAcceptTcpClient(new AsyncCallback(OnClientConnect), null);
    }

    private void HandleMessage(string message)
    {
        message = message.Trim().ToLower();

        // Orientation commands: "rx 10", "ry -20", "rz 90"
        if (message.StartsWith("rx") || message.StartsWith("ry") || message.StartsWith("rz"))
        {
            string[] parts = message.Split(' ');
            if (parts.Length == 2 && float.TryParse(parts[1], out float angle))
            {
                if (message.StartsWith("rx")) rx = angle;
                else if (message.StartsWith("ry")) ry = angle;
                else if (message.StartsWith("rz")) rz = angle;

                Debug.Log($"Orientation set: rx={rx}, ry={ry}, rz={rz} (deg)");
            }
            else
            {
                Debug.Log("Invalid command. Usage: 'rx <value>', 'ry <value>', or 'rz <value>'");
            }
            return;
        }

        // Discrete position commands
        switch (message)
        {
            case "left":     px -= positionStep; break;
            case "right":    px += positionStep; break;
            case "up":       py += positionStep; break;
            case "down":     py -= positionStep; break;
            case "forward":  pz += positionStep; break;
            case "backward": pz -= positionStep; break;
            default:
                Debug.Log("Invalid command received: " + message);
                break;
        }
    }

    private void CalculateIK()
    {
        // Tool direction "a" from rx/ry/rz
        float ax = Mathf.Cos(rz * Mathf.Deg2Rad) * Mathf.Cos(ry * Mathf.Deg2Rad);
        float ay = Mathf.Sin(rz * Mathf.Deg2Rad) * Mathf.Cos(ry * Mathf.Deg2Rad);
        float az = -Mathf.Sin(ry * Mathf.Deg2Rad);

        // Approximate wrist center position (p5)
        float p5x = px;
        float p5y = py;
        float p5z = pz - (L5 + L6) * az;

        // θ0 (base yaw)
        theta[0] = Mathf.Atan2(p5y, p5x);

        // θ2 (elbow) using cosine rule
        C3 = (Mathf.Pow(p5x, 2) + Mathf.Pow(p5y, 2) + Mathf.Pow(p5z - L1, 2) - Mathf.Pow(L2, 2) - Mathf.Pow(L3, 2)) / (2 * L2 * L3);
        C3 = Mathf.Clamp(C3, -1, 1);

        // Elbow-up solution; for elbow-down use -sqrt(...)
        theta[2] = Mathf.Atan2(Mathf.Sqrt(1 - Mathf.Pow(C3, 2)), C3);

        // θ1 (shoulder)
        float M = L2 + L3 * C3;
        float A = Mathf.Sqrt(p5x * p5x + p5y * p5y);
        float B = p5z - L1;
        theta[1] = Mathf.Atan2(M * A - L3 * Mathf.Sin((float)theta[2]) * B, L3 * Mathf.Sin((float)theta[2]) * A + M * B);

        // Solve wrist orientation (θ3–θ5) using end-effector direction vectors
        float C1 = Mathf.Cos((float)theta[0]);
        float C23 = Mathf.Cos((float)theta[1] + (float)theta[2]);
        float S1 = Mathf.Sin((float)theta[0]);
        float S23 = Mathf.Sin((float)theta[1] + (float)theta[2]);

        // Secondary direction "b" (approx.) from Euler angles
        float bx = Mathf.Cos(rx * Mathf.Deg2Rad) * Mathf.Sin(ry * Mathf.Deg2Rad) * Mathf.Cos(rz * Mathf.Deg2Rad) - Mathf.Sin(rx * Mathf.Deg2Rad) * Mathf.Sin(rz * Mathf.Deg2Rad);
        float by = Mathf.Cos(rx * Mathf.Deg2Rad) * Mathf.Sin(ry * Mathf.Deg2Rad) * Mathf.Sin(rz * Mathf.Deg2Rad) + Mathf.Sin(rx * Mathf.Deg2Rad) * Mathf.Cos(rz * Mathf.Deg2Rad);
        float bz = Mathf.Cos(rx * Mathf.Deg2Rad) * Mathf.Cos(ry * Mathf.Deg2Rad);

        float asx = C23 * (C1 * ax + S1 * ay) - S23 * az;
        float asy = -S1 * ax + C1 * ay;
        float asz = S23 * (C1 * ax + S1 * ay) + C23 * az;

        float bsx = C23 * (C1 * bx + S1 * by) - S23 * bz;
        float bsy = -S1 * bx + C1 * by;
        float bsz = S23 * (C1 * bx + S1 * by) + C23 * bz;

        theta[3] = Mathf.Atan2(asy, asx);
        theta[4] = Mathf.Atan2(Mathf.Cos((float)theta[3]) * asx + Mathf.Sin((float)theta[3]) * asy, asz);

        // Guard against sin(theta[4]) close to 0
        float s4 = Mathf.Sin((float)theta[4]);
        if (Mathf.Abs(s4) < 1e-6f)
        {
            theta[5] = 0.0;
        }
        else
        {
            theta[5] = Mathf.Atan2(Mathf.Cos((float)theta[3]) * bsy - Mathf.Sin((float)theta[3]) * bsx, -bsz / s4);
        }
    }

    private void ApplyJointAngles()
    {
        if (Joints == null || Joints.Length < 6) return;

        // IMPORTANT: these axes depend on your UR3e Unity rig.
        if (!double.IsNaN(theta[0])) Joints[0].localEulerAngles = new Vector3(0, (float)theta[0] * Mathf.Rad2Deg, 0);
        if (!double.IsNaN(theta[1])) Joints[1].localEulerAngles = new Vector3(0, 0, (float)theta[1] * Mathf.Rad2Deg);
        if (!double.IsNaN(theta[2])) Joints[2].localEulerAngles = new Vector3((float)theta[2] * Mathf.Rad2Deg, 0, 0);
        if (!double.IsNaN(theta[3])) Joints[3].localEulerAngles = new Vector3(0, (float)theta[3] * Mathf.Rad2Deg, 0);
        if (!double.IsNaN(theta[4])) Joints[4].localEulerAngles = new Vector3(0, 0, (float)theta[4] * Mathf.Rad2Deg);
        if (!double.IsNaN(theta[5])) Joints[5].localEulerAngles = new Vector3(0, (float)theta[5] * Mathf.Rad2Deg, 0);
    }

    private void OnApplicationQuit()
    {
        try
        {
            if (tcpClient != null)
            {
                stream?.Close();
                tcpClient.Close();
            }
            tcpListener?.Stop();
        }
        catch { /* ignore shutdown exceptions */ }
    }
}
