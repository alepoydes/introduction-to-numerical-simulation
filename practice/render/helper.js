function normalize(v) {
    l=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    return [v[0]/l, v[1]/l, v[2]/l];
}

function cross(a,b) {
    return [a[1]*b[2]-a[2]*b[1]
          ,a[2]*b[0]-a[0]*b[2]
          ,a[0]*b[1]-a[1]*b[0]]
}

function getShader(gl, id) {
    var shaderScript = document.getElementById(id);
    if (!shaderScript) { return null; }
    var str = "";
    var k = shaderScript.firstChild;
    while (k) {
        if (k.nodeType == 3) { str += k.textContent.trim(); }
        k = k.nextSibling;
    }
    if (shaderScript.type == "x-shader/x-fragment") {
        return compileShader(gl, str, gl.FRAGMENT_SHADER);
    } else if (shaderScript.type == "x-shader/x-vertex") {
        return compileShader(gl, str, gl.VERTEX_SHADER);
    }; 
    return null;
}

function compileShader(gl, src, type) {
    var shader=gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        alert(gl.getShaderInfoLog(shader));
        return null;
    }
    return shader;
}

function handleLoadedTexture(gl, texture) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texture.image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_NEAREST);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, null);
}
