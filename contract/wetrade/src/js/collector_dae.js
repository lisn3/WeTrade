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

var model_adr='QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco';
var deposit=10000;
var percentage=10;
Trade.methods.CreateContract(model_adr, deposit, percentage).send({from: col, value: 10000000000})
    .on('receipt', function(receipt){
        console.log("\n### CreateContract: gas used: ", receipt.gasUsed)
    });


Trade.events.Evt_Regst({
    filter:{},
    fromBlock: 0
}, function(error, event){})
    .on('data', function(event){
        console.log("....Listen....Receive an Registration....");
        var pid=event.returnValues.pid;
        var data_hash=event.returnValues.hashes;
        console.log("....pid and data hashes are...\n", pid, data_hash);

        const hash="0xaf230a56630ab5f081d5fd9d4c4c08c65417e0e49b680b87d287923aa6c025bd";
        var sample_hash=new Array(2);  //sample 2 blocks;
        sample_hash.fill(hash);

        Trade.methods.SampleData(pid, sample_hash).send({from: col})
        .on('receipt', function(receipt){
        console.log("\n### SampleData: gas used: ", receipt.gasUsed)
    });
})
.on('changed', function(event){
                   
})        
.on('error', console.error);


Trade.events.Evt_SubFeat({  
    filter:{},
    fromBlock: 0
}, function(error, event){})
    .on('data', function(event){
        console.log("....Listen....Receive the features....");
        var pid=event.returnValues.pid;
        var features=event.returnValues.features;
        console.log("....pid and feature address are...\n", pid, features);

        //DO the evaluation on features

})
.on('changed', function(event){
                  
})        
.on('error', console.error);


//After evaluation set a candidate
var opt=web3.eth.accounts[5];
Trade.methods.SetCandidate(opt).send({from: col})   
        .on('receipt', function(receipt){
        console.log("\n### SetCandidate: gas used: ", receipt.gasUsed)
});

//After receive candiate's review, set he as an optimal
Trade.events.Evt_SubReview({  
        filter:{pid: opt},
        fromBlock: 0
}, function(error, event){})
        .on('data', function(event){
        console.log("....Listen...The candidate submit his review, I get the session key...."); 

        Trade.methods.SetOptimal(opt).send({from:col})
        .on('receipt', function(receipt){
        console.log("\n### SetOptimal: gas used: ", receipt.gasUsed)
        });
})
.on('changed', function(event){
})        
.on('error', console.error);



//Receiving all the plain data addresses
Trade.events.Evt_ClaimReward({  
    filter:{pid: opt},
    fromBlock: 0
}, function(error, event){})
    .on('data', function(event){
        console.log("....Listen....Receive all the plain address....");

    //Close the session
})
.on('changed', function(event){
                  
})        
.on('error', console.error);

