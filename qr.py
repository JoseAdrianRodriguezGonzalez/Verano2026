import qrcode

url = "https://github.com/JoseAdrianRodriguezGonzalez/xidooTOPIC"
qr = qrcode.make(url)
qr.save("qr.png")
