from flask import Flask
from flask import Flask, request, render_template, url_for, session, jsonify, make_response

app = Flask(__name__)
 
@app.route('/')
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        rec = Photo(filename=filename, user=g.user.id)
        rec.store()
        flash("Photo saved.")
        return redirect(url_for('show', id=rec.id))
    return render_template('upload.html')

@app.route('/photo/<id>')
def show(id):
    photo = Photo.load(id)
    if photo is None:
        abort(404)
    url = photos.url(photo.filename)
    return render_template('show.html', url=url, photo=photo)

if __name__ == '__main__':
    app.run(host='192.168.1.254',port=9100,debug=True)