"""Optional loopback fetch gateway for public dataset reads on restricted hosts."""
import os
import urllib.parse
import urllib.request

def public_urlopen(request,timeout=40):
    gateway=os.environ.get('FNFLOW_PUBLIC_FETCH_GATEWAY')
    if not gateway:return urllib.request.urlopen(request,timeout=timeout)
    if isinstance(request,str):request=urllib.request.Request(request)
    target=gateway+'/?'+urllib.parse.urlencode({'url':request.full_url})
    headers={key:value for key,value in request.header_items() if key.lower() in {'range','accept-encoding','user-agent'}}
    forwarded=urllib.request.Request(target,headers=headers)
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(forwarded,timeout=timeout)
