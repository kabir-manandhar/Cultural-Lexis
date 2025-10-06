#!/bin/bash

echo "Do this file on 135 device"

ps aux | grep spartan-gpgpu169 | awk {'print $2'} | xargs kill -9

# ssh -fNT -L 8001:localhost:8001 spartan-gpgpu169
# ssh -fNT -L 8002:localhost:8002 spartan-gpgpu169
# ssh -fNT -L 8003:localhost:8003 spartan-gpgpu169
# ssh -fNT -L 8004:localhost:8004 spartan-gpgpu169
ssh -fNT -L 8005:localhost:8005 spartan-gpgpu169
ssh -fNT -L 8006:localhost:8006 spartan-gpgpu169
ssh -fNT -L 8007:localhost:8007 spartan-gpgpu169
ssh -fNT -L 8008:localhost:8008 spartan-gpgpu169
ssh -fNT -L 8009:localhost:8009 spartan-gpgpu169
ssh -fNT -L 8010:localhost:8010 spartan-gpgpu169
ssh -fNT -L 8011:localhost:8011 spartan-gpgpu169
ssh -fNT -L 8012:localhost:8012 spartan-gpgpu169