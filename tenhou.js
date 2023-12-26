class tenhourand{

    constructor(b64){
        if (b64 == null || b64.slice(0, 29) != 'mt19937ar-sha512-n288-base64,'){
            //seed没有，将MT置为undefined；并在之后getyamadice中返回空结果
            //TODO: 有部分seed为16进制数列，需要处理
            this.MT = undefined;
            return;
        }
        if (b64.slice(0, 29) == 'mt19937ar-sha512-n288-base64,')
            b64 = b64.slice(29);
        let arr = this.base64touint32(b64);
        this.MT = new MersenneTwister();
        this.MT.init_by_array(arr, arr.length);
        this.SHA512LEN = 64;
        this.RNDLEN = 144;
    }

    base64touint32(b64){
        let ustr = atob(b64);
        let arr = [];
        console.assert(ustr.length % 4 == 0, 'ustr length wrong');
        for (let i = 0; i < ustr.length; i += 4){
            let num = 0;
            for (let j = 3; j >= 0; j -- )
            num = num * 256 + ustr.charCodeAt(i + j);
            arr.push(num);
        }
        return new Uint32Array(arr);
    }

    outputarr(arr){
        let res = '';
        for (let i in arr)
            res += arr[i].toString(16) + ',';
        return res;
    }

    getyamadice(yama){
        if (this.MT == undefined){
            //没有seed，清空yama
            yama.length = 0;
            return [0, 0];
        }
        let src = [];
        for (let i = 0; i < this.RNDLEN * 2; i ++ ){
            let r = this.MT.genrand_int32();
            //for (let j = 3; j >= 0; j -- )
            for (let j = 0; j < 4; j ++ )
                src.push(parseInt(r / (1 << (j * 8))) % 256);
        }
        //console.log(this.outputarr(src));
        let rnd8 = [];
        for (let i = 0; i < this.RNDLEN * 2 * 4; i += this.SHA512LEN * 2)
            rnd8 = rnd8.concat(sha512.update(src.slice(i, i + this.SHA512LEN * 2)).array());
        let rnd = [];
        for (let i = 0; i < rnd8.length; i += 4 ){
            let num = 0;
            for (let j = 3; j >= 0; j -- )
                num = num * 256 + rnd8[i + j];
            rnd.push(num);
        }
        //console.log(this.outputarr(rnd));
        //TODO: 确认三麻牌山生成算法正确性
        let len = yama.length;
        for (let i = 0; i < len - 1; i ++ ){
            let tmp = yama[i];
            yama[i] = yama[rnd[i] % (len - i) + i];
            yama[rnd[i] % (len - i) + i] = tmp;
        }
        return [rnd[135] % 6, rnd[136] % 6];
    }

    /* 
    //sample code

    let b64 = 'mt19937ar-sha512-n288-base64,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';
    let yama = [], d1, d2;
    for (let i = 0; i < 136; i ++ )
        yama.push(i);

    let trand = new tenhourand(b64);
    [d1, d2] = trand.getyamadice(yama);
    console.log(d1, d2, yama);
    */

}
let b64 = 'mt19937ar-sha512-n288-base64,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';
let yama = [], d1, d2;
for (let i = 0; i < 136; i ++ )
    yama.push(i);

let trand = new tenhourand(b64);
[d1, d2] = trand.getyamadice(yama);
console.log(d1, d2, yama);
