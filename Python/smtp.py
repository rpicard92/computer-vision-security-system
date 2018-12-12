import smtplib
import os,email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email import Encoders
    

class EmailMessageBuilder:
    smtp = 0
    msg = 0
    fromEmail = 0
    toEmail = 0
    
    def buildMessage(self,fromEmail,toEmail,subject,body,video): 
        print 'Creating Message.'
        self.fromEmail = fromEmail
        self.toEmail = toEmail
        self.msg = MIMEMultipart()
        self.msg['From'] = fromEmail
        self.msg['To'] = toEmail
        self.msg['Subject'] = subject       
        self.msg.attach( MIMEText(body) )
        part = MIMEBase('application', "octet-stream")
        fo=open(video,"rb")
        part.set_payload(fo.read() )
        Encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename="%s"' % os.path.basename(video))
        self.msg.attach(part)
    
    def buildSMTPServer(self,server,port):
        print 'Starting SMPT on server: ' + str(server) +' port: ' + str(port) + '.'
        self.smtp = smtplib.SMTP(server,port)
    
    def sendMessage(self,username,password):
            print 'Sending Message.'
            self.smtp.ehlo()
            self.smtp.starttls()
            self.smtp.ehlo()
            self.smtp.login(username,password) #Edit
            self.smtp.sendmail(self.fromEmail, self.toEmail, self.msg.as_string())
            self.smtp.close()
            print 'Closing SMPT server.'
            
            

#email = ''
#password = ''
#subject = 'From Raspberry Pi!'
#body = 'Test Body'
#server = 'smtp.gmail.com'
#port = 587
#video = 'video.mp4'

#messageBuilder = EmailMessageBuilder()
#messageBuilder.buildMessage(email,email,subject,body,video)
#messageBuilder.buildSMTPServer(server,port)
#messageBuilder.sendMessage(email,password)
