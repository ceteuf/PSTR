# %%
import requests as rq 

# %%
url = "http://challenge01.root-me.org/web-serveur/ch68/"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:88.0) Gecko/20100101 Firefox/88.0",
}

proxies = {
  'http': 'http://10.1.0.0',

}

# resp = rq.get(url, headers=headers) # proxies=proxies
resp = rq.get(url, stream=True, headers=headers, proxies=proxies)
# %%
resp.content
# %%
resp.raw._connection.sock.getsockname()