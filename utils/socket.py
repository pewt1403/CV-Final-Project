
import socket
def init_udp_socket():
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	return sock

def send_data(sock, msg, IP, PORT):
    msg = bytes("{}".format(msg), encoding="utf8")
    sock.sendto(msg, (IP, PORT))