"""CPU tests of how the media worker fetches an http(s) URL: the name is resolved once, the address checked, and the socket made
to that address. A fake resolver answers a public address and then 127.0.0.1 (a rebinding name); no network is used.

    python3 -m unittest discover -s tools/vision -p test_media_worker_fetch.py -v
"""
import http.server
import socket
import threading
import unittest

from test_encoder_limits import load_media_worker

PUBLIC = "93.184.216.34"


def answer(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.enc = load_media_worker(100)
        self.resolved = []  # every name asked of the resolver, with the answers it will give in turn
        self.answers = {}
        self.connects = []  # every address a socket was asked for
        self.real = (socket.getaddrinfo, socket.create_connection)
        self.server = None
        self.seen_hosts = []
        socket.getaddrinfo = self.getaddrinfo
        socket.create_connection = self.create_connection

    def tearDown(self):
        socket.getaddrinfo, socket.create_connection = self.real
        if self.server:
            self.server.shutdown()
            self.server.server_close()

    def getaddrinfo(self, host, port, *args, **kw):
        if host not in self.answers:  # the stand-in's own connect to the local server
            return self.real[0](host, port, *args, **kw)
        self.resolved.append(host)
        turns = self.answers[host]
        return answer(turns.pop(0) if len(turns) > 1 else turns[0])

    def create_connection(self, address, timeout=None, source_address=None):
        self.connects.append(address)
        if self.server is None:
            raise ConnectionRefusedError("no network in this test")
        return self.real[1](("127.0.0.1", self.server.server_address[1]), timeout)

    def serve(self, handler):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self.server.server_address[1]

    def test_a_rebinding_name_is_connected_by_the_address_that_was_checked(self):
        self.answers["rebind.test"] = [PUBLIC, "127.0.0.1"]
        with self.assertRaises(OSError):  # the stand-in refuses the connection: nothing is reachable
            self.enc.fetch({"url": "http://rebind.test:8080/a.png"})
        self.assertEqual(self.connects, [(PUBLIC, 8080)], "the socket is made to the address that passed the check, never by name")
        self.assertEqual(self.resolved, ["rebind.test"], "the name is resolved once")

    def test_a_name_with_a_private_address_is_refused(self):
        self.answers["inside.test"] = ["10.1.2.3"]
        with self.assertRaisesRegex(self.enc.MediaError, "not a public address"):
            self.enc.fetch({"url": "http://inside.test/a.png"})
        self.assertEqual(self.connects, [])

    def test_the_host_header_is_kept_and_a_redirect_is_checked_the_same_way(self):
        outer = self

        class Redirect(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                outer.seen_hosts.append(self.headers["Host"])
                if self.path == "/ok":
                    self.send_response(200)
                    self.send_header("Content-Length", "3")
                    self.end_headers()
                    self.wfile.write(b"abc")
                else:
                    self.send_response(302)
                    self.send_header("Location", "http://inside.test/x")
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            def log_message(self, *args):
                pass

        self.serve(Redirect)
        self.answers["outside.test"] = [PUBLIC]
        self.answers["inside.test"] = ["127.0.0.1"]
        self.assertEqual(self.enc.fetch({"url": "http://outside.test/ok"}), b"abc")
        self.assertEqual(self.seen_hosts, ["outside.test"], "the Host header is the name, the connect is to the address")
        self.assertEqual(self.connects, [(PUBLIC, 80)])
        with self.assertRaisesRegex(self.enc.MediaError, "not a public address"):
            self.enc.fetch({"url": "http://outside.test/redirect"})
        self.assertEqual(self.connects, [(PUBLIC, 80), (PUBLIC, 80)], "the redirect's target was never connected to")


if __name__ == "__main__":
    unittest.main()
