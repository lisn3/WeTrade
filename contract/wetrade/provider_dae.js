var Web3 = require("web3")

var web3;

if (typeof web3 !== 'undefined') {
    web3 = new Web3(web3.currentProvider);
} else {
    web3 = new Web3(new Web3.providers.WebsocketProvider("http://127.0.0.1:8545"));
}
        
var contractAbi = [
    {
      "inputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "constructor"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "string",
          "name": "data_plain",
          "type": "string"
        }
      ],
      "name": "Evt_ClaimReward",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "col",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "string",
          "name": "modeladr",
          "type": "string"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "deposit",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "percentage",
          "type": "uint256"
        },
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "reward",
          "type": "uint256"
        }
      ],
      "name": "Evt_Create",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "bytes32[]",
          "name": "hashes",
          "type": "bytes32[]"
        }
      ],
      "name": "Evt_Regst",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "collector",
          "type": "address"
        },
        {
          "indexed": true,
          "internalType": "address",
          "name": "candidate",
          "type": "address"
        }
      ],
      "name": "Evt_SetCand",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "optimal",
          "type": "address"
        }
      ],
      "name": "Evt_SetOptimal",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "bytes32[]",
          "name": "sample_hash",
          "type": "bytes32[]"
        }
      ],
      "name": "Evt_SmpData",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "string",
          "name": "features",
          "type": "string"
        }
      ],
      "name": "Evt_SubFeat",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": true,
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "indexed": false,
          "internalType": "string",
          "name": "plain_sample",
          "type": "string"
        },
        {
          "indexed": false,
          "internalType": "bytes32",
          "name": "K_data",
          "type": "bytes32"
        }
      ],
      "name": "Evt_SubReview",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "bytes",
          "name": "ss",
          "type": "bytes"
        }
      ],
      "name": "printbytes",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "bytes32",
          "name": "a",
          "type": "bytes32"
        }
      ],
      "name": "printbytes32",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "uint256",
          "name": "a",
          "type": "uint256"
        }
      ],
      "name": "printlength",
      "type": "event"
    },
    {
      "anonymous": false,
      "inputs": [
        {
          "indexed": false,
          "internalType": "string",
          "name": "aa",
          "type": "string"
        }
      ],
      "name": "printstring",
      "type": "event"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "CF_deposit",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "CF_expiry",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "CF_percentage",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "CF_reward",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "P_col",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "P_opt",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "name": "P_profile",
      "outputs": [
        {
          "internalType": "string",
          "name": "adr_feature",
          "type": "string"
        },
        {
          "internalType": "bytes32",
          "name": "Key_data",
          "type": "bytes32"
        },
        {
          "internalType": "uint256",
          "name": "V_deposit",
          "type": "uint256"
        },
        {
          "internalType": "bool",
          "name": "isRegisted",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "sampled",
          "type": "bool"
        },
        {
          "internalType": "bool",
          "name": "setfeature",
          "type": "bool"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "P_tc",
      "outputs": [
        {
          "internalType": "address",
          "name": "",
          "type": "address"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "start_time",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [],
      "name": "state",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "payable": false,
      "stateMutability": "view",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "string",
          "name": "_modeladr",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "_deposit",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "_percentage",
          "type": "uint256"
        }
      ],
      "name": "CreateContract",
      "outputs": [],
      "payable": true,
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [
        {
          "internalType": "string",
          "name": "ss",
          "type": "string"
        },
        {
          "internalType": "uint256",
          "name": "one_adr_len",
          "type": "uint256"
        }
      ],
      "name": "ConvertString2Arr",
      "outputs": [
        {
          "internalType": "string[]",
          "name": "",
          "type": "string[]"
        }
      ],
      "payable": false,
      "stateMutability": "pure",
      "type": "function"
    },
    {
      "constant": true,
      "inputs": [
        {
          "internalType": "string",
          "name": "s1",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "s2",
          "type": "string"
        }
      ],
      "name": "StrCmp",
      "outputs": [
        {
          "internalType": "bool",
          "name": "",
          "type": "bool"
        }
      ],
      "payable": false,
      "stateMutability": "pure",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "bytes32[]",
          "name": "adr_data_hash",
          "type": "bytes32[]"
        }
      ],
      "name": "Registration",
      "outputs": [],
      "payable": true,
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "internalType": "bytes32[]",
          "name": "sample_hash",
          "type": "bytes32[]"
        }
      ],
      "name": "SampleData",
      "outputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "string",
          "name": "features",
          "type": "string"
        }
      ],
      "name": "SubmitFeatures",
      "outputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "address",
          "name": "pid",
          "type": "address"
        }
      ],
      "name": "SetCandidate",
      "outputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "address",
          "name": "pid",
          "type": "address"
        },
        {
          "internalType": "string",
          "name": "plain_sample",
          "type": "string"
        },
        {
          "internalType": "bytes32",
          "name": "K_data",
          "type": "bytes32"
        }
      ],
      "name": "SubmitReview",
      "outputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "address",
          "name": "pid",
          "type": "address"
        }
      ],
      "name": "SetOptimal",
      "outputs": [],
      "payable": false,
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [
        {
          "internalType": "string",
          "name": "data_plain",
          "type": "string"
        }
      ],
      "name": "ClaimReward",
      "outputs": [],
      "payable": true,
      "stateMutability": "payable",
      "type": "function"
    },
    {
      "constant": false,
      "inputs": [],
      "name": "ClaimDeposit",
      "outputs": [],
      "payable": true,
      "stateMutability": "payable",
      "type": "function"
    }
  ];

var contractaAddress = "0x781a9a62eE6f0FfCF336346e5e806e38cD805e66";
        
Trade = new web3.eth.Contract(contractAbi, contractaAddress);

var col=web3.eth.accounts[0];
var acct=web3.eth.accounts[5];

 //When contract create, register to it.        
Trade.events.Evt_Create({ 
        filter:{},
        fromBlock: 0
}, function(error, event){})
        .on('data', function(event){
        console.log("....Listen...The contract created....");
        console.log(event);

        var data_hash=new Array(20);  //Hash value of data addresses
        const hash="0xaf230a56630ab5f081d5fd9d4c4c08c65417e0e49b680b87d287923aa6c025bd";
        data_hash.fill(hash);

        Trade.methods.Registration(data_hash).send({from:acct, value:300000})
        .on('receipt', function(receipt){
        console.log("\n### Registration: gas used: ", receipt.gasUsed)
        });
})
.on('changed', function(event){
                   
})        
.on('error', console.error);


//Receiveing the sampled data address, then submit features.
Trade.events.Evt_SmpData({   
        filter:{pid: acct},
        fromBlock: 0
}, function(error, event){})
        .on('data', function(event){
        console.log("....Listen...Collector sampled address for me...."); // same results as the optional callback above
        
        //console.log();
        //Compute the sampled data features 

        Trade.methods.SubmitFeatures("QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo4uco").send({from:acct})
        .on('receipt', function(receipt){
        console.log("\n### SubmitFeatures: gas used: ", receipt.gasUsed)
        });
})
.on('changed', function(event){
})        
.on('error', console.error);


//when become the candidate, submit reviews.
Trade.events.Evt_SetCand({   
        filter:{candidate: acct},
        fromBlock: 0
}, function(error, event){})
        .on('data', function(event){
        console.log("....Listen...I become the candidate...."); 

        const plain="QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6uco";
        var plain_sample_adr="";
        var rpt=2;
        for(var j=0;j<rpt;j++){
            plain_sample_adr+=plain;
        }
        var key="0x911e77b214c504666bbd69642887f360310f2fc3a00907a1a32baa07bd0bfe41";

        Trade.methods.SubmitReview(acct, plain_sample_adr, key).send({from:acct})
        .on('receipt', function(receipt){
        console.log("\n### SubmitReview: gas used: ", receipt.gasUsed)
        });
})
.on('changed', function(event){
})        
.on('error', console.error);

//when become the optimal, claim rewards.
Trade.events.Evt_SetOptimal({   
        filter:{optimal: acct},
        fromBlock: 0
}, function(error, event){})
        .on('data', function(event){
        console.log("....Listen...I become the optimal...."); 

        const plain="QmXoypizjW3WknFiJnKLwHCnL66vedxjQkDDP1mXWo6uco";
        var data_plains="";
        var rpt=20;
        for(var j=0;j<rpt;j++){
            data_plains+=plain;
        }
        
        //Claim the reward with all my plain IPFS data address
        //Encrypt the address with collector's public key
        Trade.methods.ClaimReward(data_plains).send({from:acct})
        .on('receipt', function(receipt){
        console.log("\n### ClaimReward: gas used: ", receipt.gasUsed)
        });
})
.on('changed', function(event){
})        
.on('error', console.error);


Trade.methods.ClaimDeposit().send({from:acct})
.on('receipt', function(receipt){
    console.log("\n### ClaimDeposit: gas used: ", receipt.gasUsed)
});