#!/bin/bash

cat business.json| head -1000 > business_test.json
cat user.json| head -1000 > user_test.json
cat checkin.json| head -1000 > checkin_test.json
cat review.json| head -1000 > review_test.json

