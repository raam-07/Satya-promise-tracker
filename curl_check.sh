#!/bin/bash
# Script to verify status of the 20 source URLs in promises.json

echo "Checking the 20 core source URLs..."
echo "------------------------------------"

urls=(
  "p001 | Narendra Modi - Double farmer income | https://indianexpress.com/article/india/will-double-farmers-income-by-2022-says-pm-modi-4617098/"
  "p002 | Narendra Modi - 2 crore jobs | https://web.archive.org/web/20190208201614/http://www.bjp.org:80/manifesto2014"
  "p003 | Narendra Modi - Rs 15 lakh | http://lib.bjplibrary.org/jspui/handle/123456789/252"
  "p004 | Narendra Modi - 2 crore pucca houses | https://pib.gov.in/newsite/PrintRelease.aspx?relid=122709"
  "p005 | Narendra Modi - 5 trillion economy | https://pib.gov.in/PressReleasePage.aspx?PRID=1579884"
  "p006 | Narendra Modi - End corruption | http://lib.bjplibrary.org/jspui/handle/123456789/252"
  "p007 | Narendra Modi - Electoral bonds | https://pib.gov.in/PressReleasePage.aspx?PRID=1515234"
  "p008 | Narendra Modi - Har ghar nal se jal | https://pib.gov.in/PressReleasePage.aspx?PRID=1582236"
  "p009 | Narendra Modi - UCC implementation | http://library.bjp.org/jspui/handle/123456789/2988"
  "p010 | Narendra Modi - One nation one election | https://pib.gov.in/PressReleasePage.aspx?PRID=2055998"
  "p011 | Narendra Modi - Demonetization | https://pib.gov.in/newsite/PrintRelease.aspx?relid=153400"
  "p012 | Narendra Modi - Bullet train | https://pib.gov.in/PressReleasePage.aspx?PRID=1502446"
  "p013 | Amit Shah - Remove Rohingyas | https://pib.gov.in/PressReleasePage.aspx?PRID=1947234"
  "p014 | Arvind Kejriwal - World class schools | https://aamaadmiparty.org/delhi-manifesto-2015"
  "p015 | Arvind Kejriwal - Free electricity | https://aamaadmiparty.org/delhi-manifesto-2015"
  "p016 | Arvind Kejriwal - Clean Yamuna | https://aamaadmiparty.org/delhi-manifesto-2020"
  "p017 | Rahul Gandhi - NYAY scheme | https://web.archive.org/web/20190402123456/https://manifesto.inc.in/pdf/english.pdf"
  "p018 | Yogi Adityanath - End crime | https://timesofindia.indiatimes.com/assembly-elections-2017/uttar-pradesh/bjp-up-manifesto-2017-key-highlights-of-lok-kalyan-sankalp-patra/articleshow/56832269.cms"
  "p019 | Mamata Banerjee - Ma Mati Manush | https://timesofindia.indiatimes.com/assembly-elections-2011/west-bengal/trinamool-congress-manifesto-highlights/articleshow/7786440.cms"
  "p020 | Narendra Modi - Viksit Bharat | https://pib.gov.in/PressReleasePage.aspx?PRID=1852067"
)

# Set User-Agent to avoid generic CDN blocks
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

for item in "${urls[@]}"; do
  pid=$(echo "$item" | cut -d'|' -f1 | xargs)
  desc=$(echo "$item" | cut -d'|' -f2 | xargs)
  url=$(echo "$item" | cut -d'|' -f3 | xargs)
  
  # Fetch HTTP status code
  status=$(curl -A "$UA" -s -I -L -o /dev/null -w "%{http_code}" --connect-timeout 8 "$url")
  
  echo -e "$pid | Status: $status | $desc | $url"
done
