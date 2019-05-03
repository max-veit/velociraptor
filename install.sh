#!/bin/bash

if [ ! -d lib ];then mkdir lib;fi
if [ ! -d lib/SOAPFAST ];then cd lib;git clone https://github.com/dilkins/SOAPFAST.git;cd ../;fi
cd lib/SOAPFAST/soapfast;make;cd ../../../
